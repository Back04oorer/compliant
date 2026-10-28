"""IK compatibility for the source-frozen SoftMimic evaluator; no RL environment."""
from .kinematics import PiperKinematics, Q_REF


class PiperComplianceEnv(PiperKinematics):
    def close(self):
        """IK owns no renderer or simulation resources to release."""
