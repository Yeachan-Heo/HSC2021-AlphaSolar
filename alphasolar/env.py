import gym

class AlphaSolarEnv(gym.Env):
    def __init__(self, env_config):
        self.config = env_config

    def _get_observation(self) -> np.ndarray:
        """
        16x16 혹은 32x32 행렬 리턴
        구름, 태양의 위치 등이 포함되어 있는
        패널 시점의 하늘 모습이 담긴 행렬 리턴
        """
        raise NotImplementedError

    def _get_reward(self) -> np.ndarray:
        """
        채산성 공식에 따라 리워드 계산
        """
        raise NotImplementedError
    
    def _apply_action(self, action: float) -> None:
        """
        들어온 액션을 적용(패널 움직이기) 후 다음 스텝으로 옮기기
        """
        raise NotImplementedError

    def _get_terminal(self) -> bool:
        """
        환경이 마지막 스텝(마지막 시간)에 도달했는지 판단하는 함수
        환경이 마지막 스텝이라면 True 리턴, 아니라면 False 리턴
        """
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """
        환경 초기화 함수
        """
        return self._get_observation()
    
    def step(self, action):
        self._apply_action(action)
        reward = self._get_reward()
        observation = self._get_observation()

        return observation, reward, done

