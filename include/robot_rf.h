// robot_rf.h - 随机森林核心接口
// 决策树节点：label >= 0 表示叶子节点，否则按 feature/threshold 继续分裂
// 随机森林：由多棵决策树组成，通过投票得到最终预测结果
#ifndef ROBOT_RF_H
#define ROBOT_RF_H

// 决策树节点结构体
typedef struct TreeNode {
    int feature;              // 分裂特征索引，叶子节点中不使用
    double threshold;         // 分裂阈值
    int label;                // 叶子节点的类别标签，内部节点为 -1
    struct TreeNode *left;    // sample[feature] <= threshold 时进入左子树
    struct TreeNode *right;   // sample[feature] > threshold 时进入右子树
} TreeNode;

// 随机森林结构体
typedef struct {
    TreeNode **trees;         // 决策树指针数组
    int n_trees;              // 森林中树的数量
    int max_depth;            // 单棵树允许的最大深度
    int n_features;           // 每个样本的特征数量
    int n_classes;            // 分类类别数量
    int min_samples_split;    // 内部节点继续分裂所需的最小样本数
    int min_samples_leaf;     // 叶子节点允许的最小样本数
    int max_features;         // 每次分裂随机选择的特征数
} RandomForest;

// 随机森林 API：创建模型
RandomForest *rf_create(int n_trees, int max_depth, int n_features, int n_classes,
                         int min_samples_split, int min_samples_leaf);
// 随机森林 API：训练模型
void rf_train(RandomForest *rf, double **X, int *y, int n_samples);
// 随机森林 API：返回单个样本的最终预测类别
int rf_predict(RandomForest *rf, double *sample);
// 随机森林 API：返回单个样本属于每个类别的概率，调用者需要 free
double *rf_predict_proba(RandomForest *rf, double *sample);

// 决策树 API：递归构建单棵决策树
TreeNode *tree_build(double **X, int *y, int n_samples, int n_features, int n_classes,
                      int depth, int max_depth, int min_samples_split, int min_samples_leaf,
                      int max_features);
// 决策树 API：使用单棵树预测一个样本
int tree_predict(TreeNode *node, double *sample);
// 决策树 API：释放单棵树内存
void tree_free(TreeNode *node);
// 随机森林 API：释放整个森林内存
void rf_free(RandomForest *rf);

// 模型持久化：保存模型到文件
int rf_save(RandomForest *rf, const char *filename);
// 模型持久化：从文件加载模型
RandomForest *rf_load(const char *filename);

#endif
