/*
 * 机器人决策森林 配置头文件
 * 传感器: dist_front, dist_left, dist_right, speed
 * 动作: 0=FORWARD, 1=LEFT, 2=RIGHT, 3=STOP
 */
#ifndef CONFIG_H
#define CONFIG_H

#define MAX_DEPTH 10          /* 决策树最大深度 */
#define N_TREES 50            /* 随机森林树的数量 */
#define MIN_SAMPLES_SPLIT 2   /* 节点最小分裂样本数 */
#define MIN_SAMPLES_LEAF 1    /* 叶子节点最小样本数 */

#define N_FEATURES 4          /* 传感器特征数量 */
#define N_CLASSES 4           /* 动作类别数量 */

#endif
