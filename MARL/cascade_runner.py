"""
CascadeRunner: CDQN-style autoregressive DQN 的独立训练循环 (成果3/成果5)。

单进程、transition 级、off-policy。完全绕开 QMIX 的 episode ReplayBuffer /
RNN / 多进程 Pool，只复用 env（DynamicSignalEnv）与火灾信息。

指标 CSV（args.save_path/historydata/）:
- cascade_train_metrics.csv : 每个训练 episode（含探索）
- cascade_eval_metrics.csv  : 每 eval_every 个 episode 的贪心评估
列: episode, env_step, completion, evac_steps, reward_sum, fire_cum, cong_cum,
    static_cum, epsilon, loss, illegal
其中 completion=1 表示在 episode_limit 内真正疏散完成（env.done=True）。
"""
import csv
import os
import random
import sys
import tempfile

import numpy as np
import torch

from MARL.policy.cascade_dqn import CascadeDQN
from MARL.common.transition_buffer import TransitionBuffer
from utils.CTM.utils import load_all_firebytime


class CascadeRunner:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.episode_limit = args.episode_limit

        self.fire = load_all_firebytime()
        self.policy = CascadeDQN(args)
        self.buffer = TransitionBuffer(args.buffer_size, args.state_shape, self.n_agents)
        self.start_env_step = 0
        self.start_episode_idx = 0
        self.last_checkpoint_grad = 0
        self.wandb_resume_id = None
        self._pending_rng_state = None

        self.data_dir = args.save_path + '/historydata'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self._train_csv = self.data_dir + '/cascade_train_metrics.csv'
        self._eval_csv = self.data_dir + '/cascade_eval_metrics.csv'
        self._loss_file = self.data_dir + '/loss.txt'
        header = ['episode', 'env_step', 'completion', 'evac_steps', 'reward_sum',
                  'fire_cum', 'cong_cum', 'static_cum', 'epsilon', 'loss', 'illegal']
        for path in (self._train_csv, self._eval_csv):
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                with open(path, 'w', newline='') as f:
                    csv.writer(f).writerow(header)

        if getattr(args, 'resume_checkpoint', None):
            self._load_checkpoint(args.resume_checkpoint)

        self.wandb_run = None
        if getattr(args, 'wandb', False):
            self._init_wandb()
        self._restore_rng_state()

    # ---- helpers ----
    def _reset(self):
        self.env.reset(self.fire)
        mask = [self.env.get_avail_agent_actions(i) for i in range(self.n_agents)]
        self.policy.set_avail_mask(mask)
        self._mask = np.array(mask)
        return self.env.get_state()

    def _epsilon(self, env_step):
        e0, emin, span = self.args.epsilon, self.args.min_epsilon, self.args.anneal_env_steps
        return max(emin, e0 - (e0 - emin) * env_step / span)

    def _count_illegal(self, actions):
        return int(sum(1 for i, a in enumerate(actions) if self._mask[i, int(a)] == 0))

    def _write_row(self, path, row):
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def _checkpoint_signature(self):
        args = self.args
        return {
            'n_agents': self.n_agents,
            'n_actions': self.n_actions,
            'state_shape': args.state_shape,
            'buffer_size': args.buffer_size,
            'state_hidden': args.state_hidden,
            'agent_emb_dim': args.agent_emb_dim,
            'action_emb_dim': args.action_emb_dim,
            'q_hidden': args.q_hidden,
            'gamma': float(args.gamma),
            'lr': float(args.lr),
            'batch_size': int(args.batch_size),
            'warmup': int(args.warmup),
            'train_every': int(args.train_every),
            'target_update': int(args.target_update),
            'grad_norm_clip': float(args.grad_norm_clip),
            'epsilon': float(args.epsilon),
            'min_epsilon': float(args.min_epsilon),
            'anneal_env_steps': int(args.anneal_env_steps),
            'reward_norm': bool(args.reward_norm),
            'p_ref': float(args.p_ref),
        }

    def _save_checkpoint(self, env_step, episode_idx):
        model_dir = os.path.join(self.args.save_path, 'model')
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, 'latest_checkpoint.pt')
        temp_path = path + '.tmp'
        wandb_run = getattr(self, 'wandb_run', None)
        wandb_id = wandb_run.id if wandb_run is not None else self.wandb_resume_id
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        payload = {
            'format_version': 1,
            'signature': self._checkpoint_signature(),
            'policy': self.policy.training_state_dict(),
            'replay_buffer': self.buffer.state_dict(),
            'runner': {
                'env_step': int(env_step),
                'episode_idx': int(episode_idx),
            },
            'rng_state': rng_state,
            'wandb_run_id': wandb_id,
        }
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
        self.last_checkpoint_grad = self.policy.grad_steps
        return path

    def _load_checkpoint(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError('Resume checkpoint not found: {}'.format(path))
        try:
            checkpoint = torch.load(path, map_location=self.policy.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.policy.device)
        if checkpoint.get('format_version') != 1:
            raise ValueError('Unsupported checkpoint format: {}'.format(checkpoint.get('format_version')))

        expected = self._checkpoint_signature()
        actual = checkpoint.get('signature', {})
        mismatches = ['{}: {} != {}'.format(key, actual.get(key), value)
                      for key, value in expected.items() if actual.get(key) != value]
        if mismatches:
            raise ValueError('Checkpoint configuration mismatch: ' + '; '.join(mismatches))

        self.policy.load_training_state_dict(checkpoint['policy'])
        self.buffer.load_state_dict(checkpoint['replay_buffer'])
        runner_state = checkpoint['runner']
        self.start_env_step = int(runner_state['env_step'])
        self.start_episode_idx = int(runner_state['episode_idx'])
        self.last_checkpoint_grad = self.policy.grad_steps
        self.wandb_resume_id = checkpoint.get('wandb_run_id')
        self._pending_rng_state = checkpoint.get('rng_state')
        print('Resumed checkpoint: {} (env_step={}, episode={}, grad_steps={}, replay={})'.format(
            path, self.start_env_step, self.start_episode_idx,
            self.policy.grad_steps, self.buffer.size))

    def _restore_rng_state(self):
        state = self._pending_rng_state
        if not state:
            return
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if torch.cuda.is_available():
            for index, cuda_state in enumerate(state.get('cuda', [])[:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(cuda_state.cpu(), index)
        self._pending_rng_state = None

    def _init_wandb(self):
        runtime_dir = os.path.abspath(os.path.join(self.args.save_path, 'wandb_runtime'))
        runtime_paths = {
            'WANDB_DIR': os.path.join(runtime_dir, 'logs'),
            'WANDB_CACHE_DIR': os.path.join(runtime_dir, 'cache'),
            'WANDB_CONFIG_DIR': os.path.join(runtime_dir, 'config'),
            'WANDB_DATA_DIR': os.path.join(runtime_dir, 'data'),
            'WANDB_ARTIFACT_DIR': os.path.join(runtime_dir, 'artifacts'),
        }
        for env_name, path in runtime_paths.items():
            os.makedirs(path, exist_ok=True)
            os.environ.setdefault(env_name, path)

        temp_dir = os.path.join(runtime_dir, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        for env_name in ('TMPDIR', 'TMP', 'TEMP'):
            os.environ.setdefault(env_name, temp_dir)
        tempfile.tempdir = temp_dir

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError('W&B is enabled but not installed: pip install "wandb>=0.22.3,<0.23"') from exc

        args = self.args
        run_name = args.wandb_run_name or '{}-{}-{}'.format(
            args.alg, args.map, os.path.basename(os.path.normpath(args.save_path)))
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            dir=args.save_path,
            mode=args.wandb_mode,
            force=args.wandb_mode == 'online',
            job_type='train',
            allow_val_change=True,
        )
        if self.wandb_resume_id and args.wandb_mode == 'online':
            init_kwargs.update(id=self.wandb_resume_id, resume='must')
        self.wandb_run = wandb.init(**init_kwargs)
        self.wandb_run.define_metric('env_step')
        self.wandb_run.define_metric('train/*', step_metric='env_step')
        self.wandb_run.define_metric('eval/*', step_metric='env_step')
        if args.wandb_mode == 'online' and self.wandb_run.url:
            print('W&B live run:', self.wandb_run.url)

    def _log_wandb(self, prefix, env_step, metrics):
        if self.wandb_run is None:
            return
        payload = {'env_step': env_step}
        payload.update({'{}/{}'.format(prefix, key): value for key, value in metrics.items()})
        self.wandb_run.log(payload)

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    # ---- greedy evaluation (no exploration, no buffer) ----
    def evaluate(self, episode_idx, env_step):
        s = self._reset()
        fire_cum = cong_cum = static_cum = reward_sum = 0.0
        illegal = 0
        done = False
        step = 0
        while not done and step < self.episode_limit:
            a, _ = self.policy.select_actions(s, epsilon=0.0)
            illegal += self._count_illegal(a)
            r, done, comp = self.env.step([int(x) for x in a])
            static_cum += comp[0]; fire_cum += comp[1]; cong_cum += comp[2]
            reward_sum += r
            s = self.env.get_state()
            step += 1
        row = [episode_idx, env_step, int(done), step, round(reward_sum, 4),
               round(fire_cum, 4), round(cong_cum, 4), round(static_cum, 4), 0.0, '', illegal]
        self._write_row(self._eval_csv, row)
        self._log_wandb('eval', env_step, {
            'completion': int(done),
            'evac_steps': step,
            'reward_sum': reward_sum,
            'fire_cum': fire_cum,
            'cong_cum': cong_cum,
            'static_cum': static_cum,
            'illegal': illegal,
        })
        return reward_sum, int(done), step

    # ---- main training loop ----
    def run(self, num=0):
        args = self.args
        env_step = self.start_env_step
        episode_idx = self.start_episode_idx
        if env_step >= args.total_env_steps:
            raise ValueError('total_env_steps ({}) must be greater than resumed env_step ({})'.format(
                args.total_env_steps, env_step))
        last_loss = ''
        s = self._reset()
        ep_step = 0
        ep_reward = ep_fire = ep_cong = ep_static = 0.0
        ep_illegal = 0

        # Reach the requested budget, then finish the current episode so every
        # checkpoint resumes cleanly from an episode boundary.
        while env_step < args.total_env_steps or ep_step > 0:
            eps = self._epsilon(env_step)
            a, _ = self.policy.select_actions(s, eps)
            ep_illegal += self._count_illegal(a)
            r, done, comp = self.env.step([int(x) for x in a])
            s_next = self.env.get_state()

            # B3: 到 episode_limit 是时间截断，不是 terminal；bootstrap 用真实 env.done
            self.buffer.push(s, a, r, s_next, done)

            ep_static += comp[0]; ep_fire += comp[1]; ep_cong += comp[2]
            ep_reward += r
            env_step += 1
            ep_step += 1
            s = s_next

            # 训练
            if self.buffer.size >= args.warmup and env_step % args.train_every == 0:
                loss = self.policy.learn(self.buffer.sample(args.batch_size))
                last_loss = loss
                with open(self._loss_file, 'a') as f:
                    f.write(str(loss) + '\n')
                if self.policy.grad_steps % args.save_cycle == 0:
                    self.policy.save_model(self.policy.grad_steps)

            # episode 结束（真正完成 或 时间截断）
            if done or ep_step >= self.episode_limit:
                row = [episode_idx, env_step, int(done), ep_step, round(ep_reward, 4),
                       round(ep_fire, 4), round(ep_cong, 4), round(ep_static, 4),
                       round(eps, 4), last_loss, ep_illegal]
                self._write_row(self._train_csv, row)
                metrics = {
                    'completion': int(done),
                    'evac_steps': ep_step,
                    'reward_sum': ep_reward,
                    'fire_cum': ep_fire,
                    'cong_cum': ep_cong,
                    'static_cum': ep_static,
                    'epsilon': eps,
                    'illegal': ep_illegal,
                    'grad_steps': self.policy.grad_steps,
                    'buffer_size': self.buffer.size,
                }
                if last_loss != '':
                    metrics['loss'] = last_loss
                self._log_wandb('train', env_step, metrics)

                if episode_idx % max(1, args.log_every_episodes) == 0:
                    sys.stdout.write(
                        '\rcascade ep {}, env_step {}, eps {:.3f}, r {:.1f}, done {}, loss {}   '.format(
                            episode_idx, env_step, eps, ep_reward, int(done), last_loss))
                    sys.stdout.flush()

                # 每 20 个 episode 做一次贪心评估
                if episode_idx > 0 and episode_idx % 20 == 0:
                    self.evaluate(episode_idx, env_step)

                episode_idx += 1
                ep_step = 0
                ep_reward = ep_fire = ep_cong = ep_static = 0.0
                ep_illegal = 0

                if self.policy.grad_steps - self.last_checkpoint_grad >= args.save_cycle:
                    checkpoint_path = self._save_checkpoint(env_step, episode_idx)
                    print('\ncheckpoint ->', checkpoint_path)

                if env_step >= args.total_env_steps:
                    break
                s = self._reset()

        # 收尾：轻量模型、完整可恢复 checkpoint、贪心评估。
        self.policy.save_model('final')
        checkpoint_path = self._save_checkpoint(env_step, episode_idx)
        self.evaluate(episode_idx, env_step)
        print('\ncascade_dqn training done. metrics ->', self.data_dir)
        print('resume checkpoint ->', checkpoint_path)
