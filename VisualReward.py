# visual_reward.py — обёртка
import jax, jax.numpy as jnp
from flax.training import checkpoints
from serl_launcher.networks.reward_classifier import create_classifier

class VisualReward:
    def __init__(self, ckpt_dir, sample_observations, classifier_keys):
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        self.classifier = create_classifier(
            key,
            sample_observations,
            classifier_keys,
        )
        self.classifier = checkpoints.restore_checkpoint(ckpt_dir, self.classifier)
        self.classifier_keys = classifier_keys

        @jax.jit
        def _forward(params, observations):
            logits = self.classifier.apply_fn({"params": params}, observations, train=False)
            return jax.nn.sigmoid(logits)
        self._forward = _forward

    def __call__(self, obs_dict) -> float:
        # obs_dict: {"cam_front": np.uint8[H,W,3], "cam_side": ...}
        obs = {k: jnp.asarray(obs_dict[k][None, ...]) for k in self.classifier_keys}
        prob = self._forward(self.classifier.params, obs)
        return float(prob.squeeze())

    def is_success(self, obs_dict, thr=0.9) -> bool:
        return self(obs_dict) >= thr
