import math
import random
import numpy as np

# --- BEZIER DYNAMICS HUB ---

class POHModel:
    SERVICE_CEILING = 14000.0
    @staticmethod
    def get_climb_limit_fps(alt):
        limit = 11.6 * (1.0 - (alt / 18000.0))
        return max(2.0, limit)
    @staticmethod
    def get_bank_limit_deg(alt):
        return 60.0

class SkillType:
    CRUISE = "Cruise"
    PITCH = "Pitch"
    ROLL = "Roll"
    DYNAMIC = "Combined"
    ADVANCED = "Advanced"
    TACTICAL = "Tactical"

class QuinticBezier:
    @staticmethod
    def get_pos(t, p):
        return (1-t)**5 * p[0] + 5*(1-t)**4 * t * p[1] + 10*(1-t)**3 * t**2 * p[2] + \
               10*(1-t)**2 * t**3 * p[3] + 5*(1-t) * t**4 * p[4] + t**5 * p[5]
    @staticmethod
    def get_vel(t, p, D):
        return (5*(1-t)**4 * (p[1]-p[0]) + 20*(1-t)**3 * t * (p[2]-p[1]) + \
                30*(1-t)**2 * t**2 * (p[3]-p[2]) + 20*(1-t) * t**3 * (p[4]-p[3]) + \
                5*t**4 * (p[5]-p[4])) / D
    @staticmethod
    def get_acc(t, p, D):
        return (20*(1-t)**3 * (p[2]-2*p[1]+p[0]) + 60*(1-t)**2 * t * (p[3]-2*p[2]+p[1]) + \
                60*(1-t) * t**2 * (p[4]-2*p[3]+p[2]) + 20*t**3 * (p[5]-2*p[4]+p[3])) / (D**2)

class DynamicSkillGenerator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.current_skill = SkillType.CRUISE
        self.difficulty = 0.2
        self.ph, self.pa = 0.0, 5000.0
        self.vh, self.va = 0.0, 0.0
        self.ah, self.aa = 0.0, 0.0
        self.phase_p = [np.zeros(2)] * 6
        self.step_in_phase = 0
        self.phase_duration = 1000
        self.force_reset = False
        self._generate_new_phase()

    def set_skill(self, skill, difficulty=None):
        self.current_skill = skill
        if difficulty is not None: self.difficulty = difficulty
        self.force_reset = True

    def _generate_new_phase(self):
        D = self.phase_duration * self.dt
        p0 = np.array([self.ph, self.pa])
        v0 = np.array([self.vh, self.va])
        a0 = np.array([self.ah, self.aa])
        
        P0 = p0
        P1 = P0 + (v0 * D / 5.0)
        P2 = 2.0 * P1 - P0 + (a0 * D**2 / 20.0)
        
        target_h, target_a = self.ph, self.pa
        if self.current_skill == SkillType.CRUISE:
            target_a, target_h = 5000.0, 0.0
        elif self.current_skill == SkillType.PITCH:
            target_a = 5000.0 + random.uniform(-800, 800) * self.difficulty
        elif self.current_skill == SkillType.ROLL:
            target_h = self.ph + random.uniform(-90, 90) * self.difficulty
        elif self.current_skill == SkillType.DYNAMIC:
            target_a = 5000.0 + random.uniform(-800, 800) * self.difficulty
            target_h = self.ph + random.uniform(-45, 45) * self.difficulty

        P5 = np.array([target_h, target_a])
        self.phase_p = [P0, P1, P2, P5, P5, P5]
        self.step_in_phase = 0

    def peek_future_steps(self, n):
        t = min(1.0, (self.step_in_phase + n) / self.phase_duration)
        pos = QuinticBezier.get_pos(t, self.phase_p)
        return pos[0], pos[1]

    def get_next_step(self):
        if self.step_in_phase >= self.phase_duration or self.force_reset:
            self.force_reset = False
            self._generate_new_phase()
        t = self.step_in_phase / self.phase_duration
        D = self.phase_duration * self.dt
        pos = QuinticBezier.get_pos(t, self.phase_p)
        vel = QuinticBezier.get_vel(t, self.phase_p, D)
        acc = QuinticBezier.get_acc(t, self.phase_p, D)
        self.ph, self.pa, self.vh, self.va, self.ah, self.aa = pos[0], pos[1], vel[0], vel[1], acc[0], acc[1]
        self.step_in_phase += 1
        return self.ph, self.pa
