(1,2,3,4,28,5,7)
(18, 6, 12, 36, 24, 30)
[(0, [0, 0, 0, 0]), (1, [0, 0, 0, 1]), (2, [0, 0, 0, 2]), (3, [0, 0, 1, 0]), (4, [0, 0, 1, 1]), (5, [0, 0, 1, 2]), (6, [0, 1, 0, 0]),
(7, [0, 1, 0, 1]), (8, [0, 1, 0, 2]), (9, [0, 1, 1, 0]), (10, [0, 1, 1, 1]), (11, [0, 1, 1, 2]), (12, [0, 2, 0, 0]), (13, [0, 2, 0, 1]),
 (14, [0, 2, 0, 2]), (15, [0, 2, 1, 0]), (16, [0, 2, 1, 1]), (17, [0, 2, 1, 2]), (18, [1, 0, 0, 0]), (19, [1, 0, 0, 1]),
 (20, [1, 0, 0, 2]), (21, [1, 0, 1, 0]), (22, [1, 0, 1, 1]), (23, [1, 0, 1, 2]), (24, [1, 1, 0, 0]), (25, [1, 1, 0, 1]),
 (26, [1, 1, 0, 2]), (27, [1, 1, 1, 0]), (28, [1, 1, 1, 1]), (29, [1, 1, 1, 2]), (30, [1, 2, 0, 0]), (31, [1, 2, 0, 1]),
 (32, [1, 2, 0, 2]), (33, [1, 2, 1, 0]), (34, [1, 2, 1, 1]), (35, [1, 2, 1, 2]), (36, [2, 0, 0, 0]), (37, [2, 0, 0, 1]),
 (38, [2, 0, 0, 2]), (39, [2, 0, 1, 0]), (40, [2, 0, 1, 1]), (41, [2, 0, 1, 2]), (42, [2, 1, 0, 0]), (43, [2, 1, 0, 1]),
 (44, [2, 1, 0, 2]), (45, [2, 1, 1, 0]), (46, [2, 1, 1, 1]), (47, [2, 1, 1, 2]), (48, [2, 2, 0, 0]), (49, [2, 2, 0, 1]),
  (50, [2, 2, 0, 2]), (51, [2, 2, 1, 0]), (52, [2, 2, 1, 1]), (53, [2, 2, 1, 2])]

action_set_54 = {
            0: [0, 0, 0, 0],  # nop
            1: [1, 0, 0, 0],  # forward
            2: [2, 0, 0, 0],  # backward
            3: [0, 1, 0, 0],  # cam left
            4: [0, 2, 0, 0],  # cam right
            5: [1, 1, 0, 0],  # forward + cam left
            6: [2, 1, 0, 0],  # backward + cam left
            7: [1, 2, 0, 0],  # forward + cam right
            8: [2, 2, 0, 0],  # backward + cam right
            9: [0, 0, 0, 1],  # left
            10: [0, 0, 0, 2],  # right
            11: [1, 0, 0, 1],  # left + forward
            12: [2, 0, 0, 1],  # left + backward
            13: [0, 1, 0, 1],  # left + cam left
            14: [0, 2, 0, 1],  # left + cam right
            15: [1, 1, 0, 1],  # left + forward + cam left
            16: [2, 1, 0, 1],  # left + backward + cam left
            17: [1, 2, 0, 1],  # left + forward + cam right
            18: [2, 2, 0, 1],  # left + backward + cam right
            19: [1, 0, 0, 2],  # right + forward
            20: [2, 0, 0, 2],  # right + backward
            21: [0, 1, 0, 2],  # right + cam left
            22: [0, 2, 0, 2],  # right + cam right
            23: [1, 1, 0, 2],  # right + forward + cam left
            24: [2, 1, 0, 2],  # right + backward + cam left
            25: [1, 2, 0, 2],  # right + forward + cam right
            26: [2, 2, 0, 2],  # right + backward + cam right
            27: [0, 0, 1, 0],  # jump
            28: [1, 0, 1, 0],  # jump + forward
            29: [2, 0, 1, 0],  # jump + backward
            30: [0, 1, 1, 0],  # jump + cam left
            31: [0, 2, 1, 0],  # jump + cam right
            32: [1, 1, 1, 0],  # jump + forward + cam left
            33: [2, 1, 1, 0],  # jump + backward + cam left
            34: [1, 2, 1, 0],  # jump + forward + cam right
            35: [2, 2, 1, 0],  # jump + backward + cam right
            36: [0, 0, 1, 1],  # jump + left
            37: [0, 0, 1, 2],  # jump + right
            38: [1, 0, 1, 1],  # jump + left + forward
            39: [2, 0, 1, 1],  # jump + left + backward
            40: [0, 1, 1, 1],  # jump + left + cam left
            41: [0, 2, 1, 1],  # jump + left + cam right
            42: [1, 1, 1, 1],  # jump + left + forward + cam left
            43: [2, 1, 1, 1],  # jump + left + backward + cam left
            44: [1, 2, 1, 1],  # jump + left + forward + cam right
            45: [2, 2, 1, 1],  # jump + left + backward + cam right
            46: [1, 0, 1, 2],  # jump + right + forward
            47: [2, 0, 1, 2],  # jump + right + backward
            48: [0, 1, 1, 2],  # jump + right + cam left
            49: [0, 2, 1, 2],  # jump + right + cam right
            50: [1, 1, 1, 2],  # jump + right + forward + cam left
            51: [2, 1, 1, 2],  # jump + right + backward + cam left
            52: [1, 2, 1, 2],  # jump + right + forward + cam right
            53: [2, 2, 1, 2],  # jump + right + backward + cam rght
        }


import numpy as np
# from itertools import permutations
# meanings = [
#     [None, 'forward', 'backward'],    # 0 | Movement Forward/Back
#     [None, 'cam-left', 'cam-right'],  # 1 | Camera
#     [None, 'jump'],                   # 2 | Jump
#     [None, 'right', 'left'],          # 3 | Movement Left/Right
# ]

# action_meanings = [m if m else 'nothing' for m in [
#     '+'.join(m[i] for i, m in zip(self._discrete_to_vec[n], self._meanings) if m[i])
#     for n in range(54)
# ]]
[1, 0, 0, 0],[2, 0, 0, 0],[1, 1, 0, 0],[1, 2, 0, 0],[0, 0, 0, 1],[0, 0, 0, 2],[1, 0, 0, 1]
        action_set_20_limit_backwards_and_jump = {
            0: [0, 0, 0, 0],  # nop
            1: [1, 0, 0, 0],  # forward
            2: [2, 0, 0, 0],  # backward
            3: [0, 1, 0, 0],  # cam left
            4: [0, 2, 0, 0],  # cam right
            5: [1, 1, 0, 0],  # forward + cam left
            6: [1, 2, 0, 0],  # forward + cam right
            7: [0, 0, 0, 1],  # left
            8: [0, 0, 0, 2],  # right
            9: [1, 0, 0, 1],  # left + forward
            10: [0, 1, 0, 1],  # left + cam left
            11: [0, 2, 0, 1],  # left + cam right
            12: [1, 1, 0, 1],  # left + forward + cam left
            13: [1, 2, 0, 1],  # left + forward + cam right
            14: [1, 0, 0, 2],  # right + forward
            15: [0, 1, 0, 2],  # right + cam left
            16: [0, 2, 0, 2],  # right + cam right
            17: [1, 1, 0, 2],  # right + forward + cam left
            18: [1, 2, 0, 2],  # right + forward + cam right
            19: [1, 0, 1, 0],  # jump + forward
        }


dicts = [(0, [0, 0, 0, 0]), (1, [0, 0, 0, 1]), (2, [0, 0, 0, 2]), (3, [0, 0, 1, 0]), (4, [0, 0, 1, 1]), (5, [0, 0, 1, 2]), (6, [0, 1, 0, 0]), (7, [0, 1, 0, 1]), (8, [0, 1, 0, 2]), (9, [0, 1, 1, 0]), (10, [0, 1, 1, 1]), (11, [0, 1, 1, 2]), (12, [0, 2, 0, 0]), (13, [0, 2, 0, 1]), (14, [0, 2, 0, 2]), (15, [0, 2, 1, 0]), (16, [0, 2, 1, 1]), (17, [0, 2, 1, 2]), (18, [1, 0, 0, 0]), (19, [1, 0, 0, 1]), (20, [1, 0, 0, 2]), (21, [1, 0, 1, 0]), (22, [1, 0, 1, 1]), (23, [1, 0, 1, 2]), (24, [1, 1, 0, 0]), (25, [1, 1, 0, 1]), (26, [1, 1, 0, 2]), (27, [1, 1, 1, 0]), (28, [1, 1, 1, 1]), (29, [1, 1, 1, 2]), (30, [1, 2, 0, 0]), (31, [1, 2, 0, 1]), (32, [1, 2, 0, 2]), (33, [1, 2, 1, 0]), (34, [1, 2, 1, 1]), (35, [1, 2, 1, 2]), (36, [2, 0, 0, 0]), (37, [2, 0, 0, 1]), (38, [2, 0, 0, 2]), (39, [2, 0, 1, 0]), (40, [2, 0, 1, 1]), (41, [2, 0, 1, 2]), (42, [2, 1, 0, 0]), (43, [2, 1, 0, 1]), (44, [2, 1, 0, 2]), (45, [2, 1, 1, 0]), (46, [2, 1, 1, 1]), (47, [2, 1, 1, 2]), (48, [2, 2, 0, 0]), (49, [2, 2, 0, 1]), (50, [2, 2, 0, 2]), (51, [2, 2, 1, 0]), (52, [2, 2, 1, 1]), (53, [2, 2, 1, 2])]
action_names = []
# print(list(mydict.keys())[list(mydict.values()).index(16)])
actions = np.array([
            [0, 0, 0, 0],  # nop
            [1, 0, 0, 0],  # forward
            [2, 0, 0, 0],  # backward
            [0, 1, 0, 0],  # cam left
            [0, 2, 0, 0],  # cam right
            [1, 1, 0, 0],  # forward + cam left
            [1, 2, 0, 0],  # forward + cam right
            [0, 0, 0, 1],  # left
            [0, 0, 0, 2],  # right
            [1, 0, 0, 1],  # left + forward
            [0, 1, 0, 1],  # left + cam left
            [0, 2, 0, 1],  # left + cam right
            [1, 1, 0, 1],  # left + forward + cam left
            [1, 2, 0, 1],  # left + forward + cam right
            [1, 0, 0, 2],  # right + forward
            [0, 1, 0, 2],  # right + cam left
            [0, 2, 0, 2],  # right + cam right
            [1, 1, 0, 2],  # right + forward + cam left
            [1, 2, 0, 2],  # right + forward + cam right
            [1, 0, 1, 0]])
# for i in actions:
#     print(i)
#     if i in dict:
#         print(dict[i])
for search_actions in actions:
    for key, values in dict:    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if values == search_actions:
            print(key)
# for i, v in dicts:
#     print(i, v)
# _discrete_to_vec = {i: tuple(v) for i, v in dicts}
# _vec_to_discrete = {tuple(v): i for i, v in self._discrete_to_vec.items()}
#
#
# def action_vec_to_discrete(self, action: tuple):
#     action = tuple(action)
#     assert len(action) == len(self._meanings)
#     assert all(0 <= a < len(m) for a, m in zip(action, self._meanings))
#     return self._vec_to_discrete[action]
#
#     def action_discrete_to_vec(self, action: int):
#         action = int(action)
#         assert 0 <= action < len(self._discrete_to_vec)
#         return self._discrete_to_vec[action]
# counter = 0
# for i in range(3):
#     for j in range(2):
#         for k in range(3):
#             for l in range(3):
#                 counter+=1
#                 print([i,j,k,l])
# print(counter)
