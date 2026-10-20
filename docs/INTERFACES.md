# Shared Interfaces（共享接口契约）

> 冻结规则：本文档定义的接口是并行开发的硬边界。任何修改须经 Lead 审批并同步所有 Agent。单位、坐标系、状态码为规范值，不得擅自更改。

## 0. 通用约定

| 项 | 约定 |
|---|---|
| 单位 | SI 制：位置 `m`，角度 `rad`，时间 `s`，力 `N`，速度 `m/s` |
| 坐标系 | SAPIEN 世界系：`x` 前、`y` 左、`z` 上；机器人基座系：`z` 竖直向上，`x` 为臂前方；末端(tcp)系：以抓手中心为原点 |
| 姿态 | 四元数 `(w, x, y, z)`，ManiSkill 约定 |
| 状态码 | 见下方各 enum |

## 1. RobotObservation

仿真一步后机器人 + 任务的观测快照。

```python
@dataclass
class RobotObservation:
    tcp_pose: Pose        # 末端位姿（世界系，m / 四元数）
    qpos: np.ndarray      # 关节角 (rad)
    qvel: np.ndarray      # 关节速度 (rad/s)
    gripper: float        # 抓手开度 [0,1]
    task_obs: dict        # 任务相关观测（物体位姿等）
    timestamp: float      # 仿真时间 (s)
```

## 2. RobotSkill

一个可执行技能（如 pick、stack）的统一接口。

```python
class RobotSkill(ABC):
    skill_id: str                       # 唯一标识
    @abstractmethod
    def reset(self, obs: RobotObservation) -> None: ...
    @abstractmethod
    def act(self, obs: RobotObservation) -> np.ndarray:   # 返回动作（关节角/增量）
        ...
    @property
    def is_done(self) -> bool: ...
```

## 3. SkillResult

技能单次执行的返回。

```python
@dataclass
class SkillResult:
    status: SkillStatus     # 见下方 enum
    skill_id: str
    steps: int
    message: str = ""
    # 可选诊断
    final_obs: RobotObservation | None = None
    failure: FailureEvent | None = None
```

```python
class SkillStatus(Enum):
    SUCCESS = "success"        # 达到任务目标
    FAILURE = "failure"        # 执行失败（含可识别故障）
    TIMEOUT = "timeout"        # 超步数未达成
    ABORTED = "aborted"        # 被外部中止
```

## 4. SimulatorAdapter

ManiSkill 环境的统一封装，屏蔽具体 env 细节。

```python
class SimulatorAdapter(ABC):
    @abstractmethod
    def reset(self, **kw) -> RobotObservation: ...
    @abstractmethod
    def step(self, action) -> RobotObservation: ...
    @abstractmethod
    def close(self) -> None: ...
    # 元数据：任务名、最大步数、动作/观测维度、是否成功等
```

## 5. FailureEvent

运行时/仿真中检测到的故障事件（Agent B 定义，Agent C 预测）。

```python
@dataclass
class FailureEvent:
    type: FailureType     # 见下方 enum
    step: int
    timestamp: float
    detail: dict = field(default_factory=dict)   # 额外诊断

class FailureType(Enum):
    DROP = "drop"                    # 物体掉落
    COLLISION = "collision"          # 非预期碰撞
    GRIP_FAIL = "grip_fail"          # 抓取失败/滑脱
    MOTION_FAIL = "motion_fail"      # 规划/运动失败
    TIMEOUT = "timeout"              # 步数超限
    UNKNOWN = "unknown"
```

## 6. EpisodeResult

一个完整 episode 的评估结果（Agent D 定义）。

```python
@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    steps: int
    reward: float
    skill_results: list[SkillResult]
    failures: list[FailureEvent]
    metrics: dict        # 可扩展指标
```

## 7. 接口归属（防冲突）

| 接口 | 冻结由谁 | 实现由谁 |
|---|---|---|
| RobotObservation | Lead | 所有 |
| RobotSkill / SkillResult | Lead | Agent B |
| SimulatorAdapter | Lead | Agent A |
| FailureEvent | Lead | Agent B（定义）、Agent C（预测模型消费） |
| EpisodeResult | Lead | Agent D |
