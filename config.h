// config.h - 机器人决策森林配置头文件
// 传感器特征：dist_front, dist_left, dist_right, speed
// 动作类别：0=FORWARD, 1=LEFT, 2=RIGHT, 3=STOP
#ifndef CONFIG_H
#define CONFIG_H

// 决策树最大深度，限制树的复杂度，防止过拟合
#define MAX_DEPTH 10
// 随机森林中决策树的数量
#define N_TREES 50
// 节点继续分裂所需的最小样本数
#define MIN_SAMPLES_SPLIT 2
// 叶子节点允许的最小样本数
#define MIN_SAMPLES_LEAF 1

// 每条样本中的传感器特征数量
#define N_FEATURES 4
// 机器人动作类别数量
#define N_CLASSES 4

#endif
