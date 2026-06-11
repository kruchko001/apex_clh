from stable_baselines3.common.callbacks import BaseCallback

from tron_paper.training.gui_rollout import StationaryGUIWorker


class TrainGUICallback(BaseCallback):
    def __init__(self, delay_ms: int = 50, every_updates: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.delay_ms = delay_ms
        self.every_updates = every_updates
        self._updates = 0
        self._orig_train = None
        self._worker = StationaryGUIWorker(delay_ms)

    def _on_step(self) -> bool:
        return True

    def _init_callback(self) -> None:
        self._worker.start()
        self._orig_train = self.model.train

        def wrapped():
            self._orig_train()
            self._updates += 1
            if self._updates % self.every_updates == 0:
                self._worker.push(self.model, self._updates)
                if self.verbose:
                    pending = self._worker.pending()
                    print(f"GUI queued update #{self._updates} (pending={pending})")

        self.model.train = wrapped

    def _on_training_end(self) -> None:
        if self._orig_train is not None:
            self.model.train = self._orig_train
        self._worker.stop()
