/*
 * robot_rf.h - 随机森林核心接口
 *
 * 决策树节点: label>=0 为叶子节点，否则按 feature/threshold 分裂
 * 随机森林: 多棵树的 Bagging 集成，每次分裂随机选取 sqrt(n_features) 个特征
 * 序列化: 文本格式，T=节点 N=空，前序遍历
 */
#ifndef ROBOT_RF_H
#define ROBOT_RF_H

typedef struct TreeNode {
    int feature;              /* 分裂特征索引（叶节点为 -1） */
    double threshold;         /* 分裂阈值 */
    int label;                /* 叶节点类别标签，内部节点为 -1 */
    struct TreeNode *left;    /* sample[feature] <= threshold */
    struct TreeNode *right;   /* sample[feature] >  threshold */
} TreeNode;

typedef struct {
    TreeNode **trees;         /* 决策树指针数组 */
    int n_trees;              /* 森林中树的数量 */
    int max_depth;            /* 单棵树允许的最大深度 */
    int n_features;           /* 每个样本的特征数量 */
    int n_classes;            /* 分类类别数量 */
    int min_samples_split;    /* 内部节点继续分裂所需的最小样本数 */
    int min_samples_leaf;     /* 左右叶子节点允许的最小样本数 */
    int max_features;         /* 每次分裂的随机特征数 = sqrt(n_features) */
} RandomForest;

/* 随机森林 API */
RandomForest *rf_create(int n_trees, int max_depth, int n_features, int n_classes,
                         int min_samples_split, int min_samples_leaf);
/* 使用训练集 X/y 训练所有树，并在训练过程中打印 OOB 准确率 */
void rf_train(RandomForest *rf, double **X, int *y, int n_samples);
/* 返回单个样本的最终预测类别 */
int rf_predict(RandomForest *rf, double *sample);
double *rf_predict_proba(RandomForest *rf, double *sample);  /* 返回各类别概率，调用者需 free */

/* 决策树 API */
TreeNode *tree_build(double **X, int *y, int n_samples, int n_features, int n_classes,
                      int depth, int max_depth, int min_samples_split, int min_samples_leaf,
                      int max_features);
int tree_predict(TreeNode *node, double *sample);
void tree_free(TreeNode *node);
void rf_free(RandomForest *rf);

/* 模型持久化 */
int rf_save(RandomForest *rf, const char *filename);
RandomForest *rf_load(const char *filename);

#endif
