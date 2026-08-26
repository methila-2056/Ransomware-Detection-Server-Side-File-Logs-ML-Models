"""
Ransomware Detection - File Operation Simulator

Simulates realistic server-side file operations for an SME environment.
Based on the Aranyi et al. (2026) paper's methodology:
- 4 user profiles generating benign workload
- 5 ransomware families with characteristic attack patterns
- Operations aggregated per 1-second window
"""

import random
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

log = logging.getLogger("ransomware.sim")


@dataclass
class UserProfiles:
    """File operation ranges for each user profile (from paper Table 3)."""

    PROFILES = {
        "Secretary": {
            "create": (30, 60),
            "rename": (10, 20),
            "delete": (5, 20),
            "description": "Office document editing, filing",
        },
        "IT Admin": {
            "create": (30, 50),
            "rename": (10, 30),
            "delete": (10, 20),
            "description": "System management, backups",
        },
        "CEO": {
            "create": (0, 10),
            "rename": (0, 5),
            "delete": (0, 5),
            "description": "Light email and document work",
        },
        "Remote Worker": {
            "create": (20, 60),
            "rename": (10, 20),
            "delete": (5, 10),
            "description": "VPN-based file sync operations",
        },
    }


@dataclass
class RansomwareFamilies:
    """Attack patterns for 5 ransomware families."""

    FAMILIES = {
        "Ryuk": {
            "create": (80, 120),
            "rename": (40, 70),
            "delete": (30, 50),
            "duration_range": (10, 20),
            "ramp_up": 3,
            "description": "Targeted encryption, high-volume operations",
        },
        "WannaCry": {
            "create": (60, 100),
            "rename": (50, 80),
            "delete": (20, 40),
            "duration_range": (8, 15),
            "ramp_up": 2,
            "description": "Rapid spread, aggressive file operations",
        },
        "NotPetya": {
            "create": (70, 110),
            "rename": (30, 60),
            "delete": (40, 60),
            "duration_range": (12, 25),
            "ramp_up": 4,
            "description": "Destructive wiper disguised as ransomware",
        },
        "Lockbit": {
            "create": (90, 130),
            "rename": (50, 90),
            "delete": (35, 55),
            "duration_range": (5, 15),
            "ramp_up": 2,
            "description": "Fastest encryption, high throughput",
        },
        "Teslacrypt": {
            "create": (50, 90),
            "rename": (20, 50),
            "delete": (25, 45),
            "duration_range": (10, 20),
            "ramp_up": 3,
            "description": "Moderate speed, gaming file targets",
        },
    }


@dataclass
class SimulationState:
    """Tracks the current state of the simulation."""

    is_running: bool = False
    is_attack_active: bool = False
    current_user: str = "Secretary"
    current_attack_family: Optional[str] = None
    attack_seconds_remaining: int = 0
    attack_ramp_up_remaining: int = 0
    seconds_until_next_attack: int = 0
    total_seconds_simulated: int = 0
    total_attacks: int = 0
    attack_history: List[dict] = field(default_factory=list)
    simulation_speed: float = 1.0


class FileOperationSimulator:
    """
    Simulates server-side file operations for ransomware detection.

    Generates 1-second aggregated feature vectors:
    - nc: number of file creation operations
    - nr: number of file renaming operations
    - nu: number of file unlinking/deletion operations
    - att: binary attack indicator (0=benign, 1=attack)
    """

    def __init__(self, attack_interval_range=(15, 30)):
        self.user_profiles = UserProfiles.PROFILES
        self.ransomware_families = RansomwareFamilies.FAMILIES
        self.state = SimulationState()
        self.attack_interval_range = attack_interval_range
        self._reset_attack_timer()

    def _reset_attack_timer(self):
        self.state.seconds_until_next_attack = random.randint(
            self.attack_interval_range[0], self.attack_interval_range[1]
        )

    def _pick_random_user(self) -> str:
        return random.choice(list(self.user_profiles.keys()))

    def _pick_random_family(self) -> str:
        return random.choice(list(self.ransomware_families.keys()))

    def _generate_benign_ops(self, user: str) -> Tuple[int, int, int]:
        profile = self.user_profiles[user]
        nc = random.randint(*profile["create"])
        nr = random.randint(*profile["rename"])
        nu = random.randint(*profile["delete"])
        return nc, nr, nu

    def _generate_attack_ops(self, family: str, progress: float) -> Tuple[int, int, int]:
        fam = self.ransomware_families[family]
        ramp_factor = min(1.0, progress / 0.3) if fam["ramp_up"] > 0 else 1.0

        base_nc = random.randint(*fam["create"])
        base_nr = random.randint(*fam["rename"])
        base_nu = random.randint(*fam["delete"])

        nc = int(base_nc * ramp_factor)
        nr = int(base_nr * ramp_factor)
        nu = int(base_nu * ramp_factor)

        return nc, nr, nu

    def generate_next_tick(self) -> dict:
        self.state.total_seconds_simulated += 1

        result = {
            "timestamp": self.state.total_seconds_simulated,
            "nc": 0, "nr": 0, "nu": 0,
            "att": 0, "user": "", "family": None,
            "description": "", "is_attack": False,
        }

        if self.state.is_attack_active:
            self.state.attack_seconds_remaining -= 1
            total_duration = self.ransomware_families[self.state.current_attack_family]["duration_range"]
            elapsed = max(1, (total_duration[0] + total_duration[1]) // 2 - self.state.attack_seconds_remaining)
            progress = elapsed / max(1, (total_duration[0] + total_duration[1]) // 2)

            nc, nr, nu = self._generate_attack_ops(self.state.current_attack_family, progress)

            if random.random() < 0.3:
                benign_nc, benign_nr, benign_nu = self._generate_benign_ops(self.state.current_user)
                nc += benign_nc // 3
                nr += benign_nr // 3
                nu += benign_nu // 3

            result["nc"] = nc
            result["nr"] = nr
            result["nu"] = nu
            result["att"] = 1
            result["user"] = self.state.current_user
            result["family"] = self.state.current_attack_family
            result["description"] = self.ransomware_families[self.state.current_attack_family]["description"]
            result["is_attack"] = True

            if self.state.attack_seconds_remaining <= 0:
                self.state.is_attack_active = False
                self.state.current_attack_family = None
                self._reset_attack_timer()

        else:
            nc, nr, nu = self._generate_benign_ops(self.state.current_user)
            result["nc"] = nc
            result["nr"] = nr
            result["nu"] = nu
            result["att"] = 0
            result["user"] = self.state.current_user
            result["description"] = self.user_profiles[self.state.current_user]["description"]

            self.state.seconds_until_next_attack -= 1
            if self.state.seconds_until_next_attack <= 0:
                family = self._pick_random_family()
                duration_range = self.ransomware_families[family]["duration_range"]
                duration = random.randint(*duration_range)

                self.state.is_attack_active = True
                self.state.current_attack_family = family
                self.state.attack_seconds_remaining = duration
                self.state.total_attacks += 1

                self.state.attack_history.append({
                    "attack_id": self.state.total_attacks,
                    "family": family,
                    "start_second": self.state.total_seconds_simulated,
                    "duration": duration,
                })

                nc, nr, nu = self._generate_attack_ops(family, 0.0)
                result["nc"] = nc
                result["nr"] = nr
                result["nu"] = nu
                result["att"] = 1
                result["family"] = family
                result["description"] = self.ransomware_families[family]["description"]
                result["is_attack"] = True
                self.state.attack_seconds_remaining -= 1

                if self.state.attack_seconds_remaining <= 0:
                    self.state.is_attack_active = False
                    self.state.current_attack_family = None
                    self._reset_attack_timer()

            if random.random() < 0.1:
                self.state.current_user = self._pick_random_user()

        return result

    def start(self):
        self.state.is_running = True
        self.state.current_user = self._pick_random_user()
        self._reset_attack_timer()

    def stop(self):
        self.state.is_running = False

    def get_state(self) -> dict:
        return {
            "is_running": self.state.is_running,
            "is_attack_active": self.state.is_attack_active,
            "current_user": self.state.current_user,
            "current_attack_family": self.state.current_attack_family,
            "attack_seconds_remaining": self.state.attack_seconds_remaining,
            "seconds_until_next_attack": self.state.seconds_until_next_attack,
            "total_seconds_simulated": self.state.total_seconds_simulated,
            "total_attacks": self.state.total_attacks,
            "attack_history": self.state.attack_history[-10:],
            "simulation_speed": self.state.simulation_speed,
        }

    def set_speed(self, speed: float):
        self.state.simulation_speed = max(0.1, min(10.0, speed))

    def force_attack(self, family: str = None) -> bool:
        if self.state.is_attack_active:
            return False

        if family is None:
            family = self._pick_random_family()
        elif family not in self.ransomware_families:
            return False

        duration_range = self.ransomware_families[family]["duration_range"]
        duration = random.randint(*duration_range)

        self.state.is_attack_active = True
        self.state.current_attack_family = family
        self.state.attack_seconds_remaining = duration
        self.state.total_attacks += 1

        self.state.attack_history.append({
            "attack_id": self.state.total_attacks,
            "family": family,
            "start_second": self.state.total_seconds_simulated,
            "duration": duration,
        })

        log.info("Forced attack: %s (duration %ds)", family, duration)
        return True


def generate_training_data(num_samples: int = 10000) -> List[dict]:
    """Generate synthetic training data for ML model development."""
    data = []

    benign_count = 0
    attack_count = 0
    target_per_class = num_samples // 2

    users = list(UserProfiles.PROFILES.keys())

    idle_samples = target_per_class // 10
    for _ in range(idle_samples):
        data.append({"nc": 0, "nr": 0, "nu": 0, "att": 0})
        benign_count += 1

    while benign_count < target_per_class:
        user = random.choice(users)
        profile = UserProfiles.PROFILES[user]
        nc = random.randint(*profile["create"])
        nr = random.randint(*profile["rename"])
        nu = random.randint(*profile["delete"])

        if random.random() < 0.05:
            nc = int(nc * random.uniform(1.5, 2.5))
            nr = int(nr * random.uniform(1.3, 2.0))
            nu = int(nu * random.uniform(1.2, 1.8))

        data.append({"nc": nc, "nr": nr, "nu": nu, "att": 0})
        benign_count += 1

    families = list(RansomwareFamilies.FAMILIES.keys())
    while attack_count < target_per_class:
        family = random.choice(families)
        fam = RansomwareFamilies.FAMILIES[family]
        duration = random.randint(*fam["duration_range"])

        for second in range(duration):
            if attack_count >= target_per_class:
                break

            progress = second / max(1, duration)
            ramp_factor = min(1.0, progress / 0.3) if fam["ramp_up"] > 0 else 1.0
            min_activity = max(3, int(ramp_factor * 10))

            nc = int(random.randint(*fam["create"]) * ramp_factor)
            nr = int(random.randint(*fam["rename"]) * ramp_factor)
            nu = int(random.randint(*fam["delete"]) * ramp_factor)

            nc = max(nc, min_activity)
            nr = max(nr, min_activity // 2)
            nu = max(nu, min_activity // 2)

            if random.random() < 0.2:
                user = random.choice(users)
                profile = UserProfiles.PROFILES[user]
                nc += random.randint(*profile["create"]) // 4
                nr += random.randint(*profile["rename"]) // 4
                nu += random.randint(*profile["delete"]) // 4

            data.append({"nc": max(1, nc), "nr": max(0, nr), "nu": max(0, nu), "att": 1})
            attack_count += 1

    transitions = max(1, num_samples // 20)
    for _ in range(transitions):
        family = random.choice(families)
        fam = RansomwareFamilies.FAMILIES[family]

        for i in range(5):
            factor = (i + 1) / 5.0
            nc = max(3, int(random.randint(*fam["create"]) * factor * 0.7))
            nr = max(1, int(random.randint(*fam["rename"]) * factor * 0.7))
            nu = max(1, int(random.randint(*fam["delete"]) * factor * 0.7))
            data.append({"nc": nc, "nr": nr, "nu": nu, "att": 1})

    random.shuffle(data)
    log.info("Generated %d training samples", len(data))
    return data


if __name__ == "__main__":
    sim = FileOperationSimulator(attack_interval_range=(5, 10))
    sim.start()

    print("Generating 30 seconds of simulated data...")
    for i in range(30):
        tick = sim.generate_next_tick()
        marker = " *** ATTACK ***" if tick["is_attack"] else ""
        print(
            f"  [{tick['timestamp']:3d}s] nc={tick['nc']:3d} nr={tick['nr']:3d} "
            f"nu={tick['nu']:3d} att={tick['att']} user={tick['user']:15s}{marker}"
        )

    print(f"\nTotal attacks: {sim.state.total_attacks}")
    print(f"Attack history: {sim.state.attack_history}")
