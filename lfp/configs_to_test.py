


IMB00003 = '''
IMB0_00003
--train_dataset UR5 UR5_slow_gripper UR5_high_transition
--test_dataset UR5_slow_gripper_test
-c
-d GPU
-b 512
-la 2048
-le 512
-lp 512
-z 256
-lr 3e-4
-i
-tfr

'''.split()

B00003 = '''
GCSB0_00003
--train_dataset UR5 UR5_slow_gripper UR5_high_transition
--test_dataset UR5_slow_gripper_test
-c
-d GPU
-b 512
-la 2048
-le 512
-lp 512
-z 256
-lr 3e-4

'''.split()


PB02 = '''
PROB0_02
--train_dataset UR5 UR5_slow_gripper UR5_high_transition
--test_dataset UR5_slow_gripper_test
-c
-d GPU
-b 512
-la 2048
-le 512
-lp 512
-z 256
-lr 3e-4
-n 5

'''.split()


B0003 = '''
GCSB0_0003
--train_dataset UR5 UR5_slow_gripper UR5_high_transition
--test_dataset UR5_slow_gripper_test
-c
-d GPU
-b 512
-la 2048
-le 512
-lp 512
-z 256
-lr 3e-4

'''.split()


B000003 = '''
GCSB0_000003
--train_dataset UR5 UR5_slow_gripper UR5_high_transition
--test_dataset UR5_slow_gripper_test
-c
-d GPU
-b 512
-la 2048
-le 512
-lp 512
-z 256
-lr 3e-4

'''.split()
