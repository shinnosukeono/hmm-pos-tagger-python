import numpy as np
from const import POS

from .model import Model


class HMM(Model):
    def __init__(self, n_pos: int, n_words: int) -> None:
        self.n_pos = n_pos
        self.n_words = n_words
        self.transition_probability: np.ndarray = np.zeros((self.n_pos, self.n_pos))
        self.emission_probability: np.ndarray = np.zeros((self.n_pos, self.n_words))
        self.transition_count: np.ndarray = np.zeros((self.n_pos, self.n_pos))
        self.emission_count: np.ndarray = np.zeros((self.n_pos, self.n_words))
        self.is_train_mode: bool = True

    def _train(self, batch_s: list, batch_i: list) -> None:
        updated_set_emission = set()
        updated_set_transition = set()
        for sentence_s, sentence_i in zip(batch_s, batch_i):
            # count the transition and the emission
            for idx, (s, i) in enumerate(zip(sentence_s, sentence_i)):
                # count the transition
                if idx == 0:  # the transition from BOS
                    self.transition_count[POS.BOS_VALUE, i] += 1
                    updated_set_transition.add(POS.BOS_VALUE)

                if idx < len(sentence_s) - 1:
                    j = sentence_i[idx + 1]
                    self.transition_count[i, j] += 1
                    updated_set_transition.add(i)

                # count the emission
                self.emission_count[i, s] += 1

                # record the updated index
                updated_set_emission.add(i)

        # update the parameter
        for i in updated_set_emission:
            sum_emission = np.sum(self.emission_count[i])
            for j in range(self.n_words):
                self.emission_probability[i, j] = self.emission_count[i, j] / sum_emission

        for i in updated_set_transition:
            sum_transition = np.sum(self.transition_count[i])
            for j in range(self.n_pos):
                self.transition_probability[i, j] = self.transition_count[i, j] / sum_transition

        # check the validity of the training
        assert np.allclose(np.sum(self.emission_probability[1:], axis=1), 1)  # ignore the row for BOS
        assert np.allclose(np.sum(self.transition_probability, axis=1), 1)

    def _viterbi(self, sentence_s: list) -> tuple:
        q: np.ndarray = np.zeros((len(sentence_s), self.n_pos))
        xi: np.ndarray = np.zeros((len(sentence_s), self.n_pos))

        def _log(x):
            return np.log(x) if x > 0 else np.log(1e-10)

        # initialize
        for k in range(self.n_pos):
            q[0, k] = _log(self.transition_probability[POS.BOS_VALUE, k]) + _log(self.emission_probability[k, sentence_s[0]])
            xi[0, k] = POS.BOS_VALUE

        # forward
        for t in range(len(sentence_s) - 1):
            s = sentence_s[t + 1]
            for k in range(self.n_pos):
                current_state_probability = np.array(
                    [q[t, i] + _log(self.transition_probability[i, k]) for i in range(self.n_pos)]
                )
                idx = np.argmax(current_state_probability)
                q[t + 1, k] = current_state_probability[idx] + _log(self.emission_probability[k, s])
                xi[t + 1, k] = idx

        idx = np.argmax(q[-1])
        q_pred = q[-1, idx]

        # backtrack
        i_pred = np.zeros(len(sentence_s), dtype=int)
        i_pred[-1] = idx
        for t in range(len(sentence_s) - 2, -1, -1):
            idx = i_pred[t + 1]
            i_pred[t] = xi[t + 1, idx]

        return q_pred, i_pred

    def _eval(self, batch_s: list) -> list:
        res = []
        for sentence_s in batch_s:
            res.append(self._viterbi(sentence_s))

        return res

    def train(self) -> None:
        self.is_train_mode = True

    def eval(self) -> None:
        self.is_train_mode = False

    def forward(self, batch_s: list, batch_i: list) -> list | None:
        if self.is_train_mode:
            self._train(batch_s, batch_i)
            return None
        else:
            return self._eval(batch_s)
