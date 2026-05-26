// robot_rf.c - 随机森林实现
// 训练流程：Bootstrap 采样 -> 构建 CART 决策树 -> 使用 OOB 样本评估
// 预测流程：每棵树独立预测 -> 多数投票得到最终类别 -> 统计概率
// 保存格式：使用前序遍历把树结构写入文本文件
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// 随机森林结构和函数声明
#include "robot_rf.h"
// 工具函数声明
#include "utils.h"

// 排序辅助结构：保存某个样本的特征值和类别标签
typedef struct {
    double value;
    int label;
} SortPair;

// qsort 比较函数：按照特征值从小到大排序
static int cmp_sort_pair(const void *a, const void *b) {
    double da = ((const SortPair *)a)->value;
    double db = ((const SortPair *)b)->value;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

// 返回样本中出现次数最多的类别，用作叶子节点的预测值
static int majority_class(int *y, int n_samples, int n_classes) {
    int *counts = (int *)calloc((size_t)n_classes, sizeof(int));
    int best = 0;
    int i;
    for (i = 0; i < n_samples; i++) counts[y[i]]++;
    int best_count = counts[0];
    for (i = 1; i < n_classes; i++) {
        if (counts[i] > best_count) {
            best_count = counts[i];
            best = i;
        }
    }
    free(counts);
    return best;
}

// 判断当前节点中的样本是否全部属于同一类别
static int all_same_class(int *y, int n) {
    int i;
    for (i = 1; i < n; i++) {
        if (y[i] != y[0]) return 0;
    }
    return 1;
}

// 在随机选出的候选特征子集中寻找最优分裂条件
// 思路：对每个特征排序样本，从左到右扫描可能的阈值，选择加权 Gini 最小的位置
static void find_best_split(double **X, int *y, int *indices, int n,
                             int n_features, int n_classes,
                             int min_samples_leaf,
                             int *best_feature, double *best_threshold,
                             int *selected_features, int n_select) {
    double best_gini = 1e9;
    int f, i;

    *best_feature = -1;
    *best_threshold = 0.0;

    // pairs 保存当前特征下的 (特征值, 标签)，排序后可以线性扫描候选阈值
    SortPair *pairs = (SortPair *)malloc((size_t)n * sizeof(SortPair));

    // 遍历候选特征
    for (f = 0; f < n_select; f++) {
        int feature = selected_features[f];

        // 把当前节点的样本拷贝到 pairs，方便按该特征值排序
        for (i = 0; i < n; i++) {
            int idx = indices[i];
            pairs[i].value = X[idx][feature];
            pairs[i].label = y[idx];
        }
        qsort(pairs, (size_t)n, sizeof(SortPair), cmp_sort_pair);

        // 初始时所有样本都在右侧，扫描过程中逐个移动到左侧
        int right_counts[16] = {0};
        for (i = 0; i < n; i++) right_counts[pairs[i].label]++;

        int left_counts[16] = {0};
        int left_n = 0;

        // 扫描相邻样本之间的分裂点
        for (i = 0; i < n - 1; i++) {
            int lbl = pairs[i].label;
            left_counts[lbl]++;
            right_counts[lbl]--;
            left_n++;

            double val = pairs[i].value;
            double next_val = pairs[i + 1].value;

            // 跳过相同值的样本对，分裂阈值必须位于两个不同特征值之间
            if (val >= next_val - 1e-10) continue;

            int right_n = n - left_n;
            // 左右子节点样本数量必须满足最小叶子节点限制
            if (left_n < min_samples_leaf || right_n < min_samples_leaf) continue;

            // 分别计算左子节点和右子节点的 Gini 不纯度
            double gini_left = 1.0;
            double gini_right = 1.0;
            int c;
            for (c = 0; c < n_classes; c++) {
                if (left_n > 0) {
                    double pl = (double)left_counts[c] / (double)left_n;
                    gini_left -= pl * pl;
                }
                if (right_n > 0) {
                    double pr = (double)right_counts[c] / (double)right_n;
                    gini_right -= pr * pr;
                }
            }

            // 用左右子节点样本占比加权，得到该阈值的整体不纯度
            double weighted_gini = ((double)left_n / n) * gini_left
                                 + ((double)right_n / n) * gini_right;

            // 记录当前找到的最优分裂特征和阈值
            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                *best_feature = feature;
                *best_threshold = (val + next_val) / 2.0;
            }
        }
    }

    free(pairs);
}

TreeNode *tree_build(double **X, int *y, int n, int n_features, int n_classes,
                      int depth, int max_depth, int min_samples_split,
                      int min_samples_leaf, int max_features) {
    // 创建当前节点，默认 label 为 -1 表示内部节点
    TreeNode *node = (TreeNode *)calloc(1, sizeof(TreeNode));
    node->label = -1;

    // 停止条件：样本不足、深度达到上限、样本太少、或节点已经纯净
    if (n < min_samples_split || depth >= max_depth || n < 2 ||
        all_same_class(y, n)) {
        // 停止分裂时，将当前节点设置为多数类叶子节点
        node->label = majority_class(y, n, n_classes);
        return node;
    }

    // 构造当前节点样本索引数组
    int *indices = (int *)malloc((size_t)n * sizeof(int));
    int i;
    for (i = 0; i < n; i++) indices[i] = i;

    // 随机打乱特征索引，再只选择前 max_features 个特征参与分裂
    int *feature_pool = (int *)malloc((size_t)n_features * sizeof(int));
    for (i = 0; i < n_features; i++) feature_pool[i] = i;
    shuffle_features(feature_pool, n_features);

    int n_select = max_features;
    if (n_select > n_features) n_select = n_features;

    int best_feature = -1;
    double best_threshold = 0.0;

    // 在随机特征子集内寻找本节点的最佳切分条件
    find_best_split(X, y, indices, n, n_features, n_classes,
                     min_samples_leaf,
                     &best_feature, &best_threshold,
                     feature_pool, n_select);

    if (best_feature < 0) {
        // 如果找不到合法分裂点，则当前节点变成多数类叶子节点
        node->label = majority_class(y, n, n_classes);
        free(indices);
        free(feature_pool);
        return node;
    }

    // 按最优特征和阈值把样本分到左子树或右子树
    int *left_idx = (int *)malloc((size_t)n * sizeof(int));
    int *right_idx = (int *)malloc((size_t)n * sizeof(int));
    int *left_y = (int *)malloc((size_t)n * sizeof(int));
    int *right_y = (int *)malloc((size_t)n * sizeof(int));
    int left_n = 0, right_n = 0;

    for (i = 0; i < n; i++) {
        int idx = indices[i];
        if (X[idx][best_feature] <= best_threshold) {
            left_idx[left_n] = idx;
            left_y[left_n] = y[idx];
            left_n++;
        } else {
            right_idx[right_n] = idx;
            right_y[right_n] = y[idx];
            right_n++;
        }
    }

    if (left_n < min_samples_leaf || right_n < min_samples_leaf) {
        // 分裂后任一侧样本太少，则放弃分裂，改为叶子节点
        node->label = majority_class(y, n, n_classes);
        free(indices);
        free(feature_pool);
        free(left_idx);
        free(right_idx);
        free(left_y);
        free(right_y);
        return node;
    }

    node->feature = best_feature;
    node->threshold = best_threshold;

    // 浅拷贝左子树的 X 指针数组，只复制行指针，不复制特征数据本身
    double **X_left = (double **)malloc((size_t)left_n * sizeof(double *));
    for (i = 0; i < left_n; i++) X_left[i] = X[left_idx[i]];

    // 浅拷贝右子树的 X 指针数组
    double **X_right = (double **)malloc((size_t)right_n * sizeof(double *));
    for (i = 0; i < right_n; i++) X_right[i] = X[right_idx[i]];

    // 递归构建左子树和右子树
    node->left = tree_build(X_left, left_y, left_n, n_features, n_classes,
                             depth + 1, max_depth, min_samples_split,
                             min_samples_leaf, max_features);
    node->right = tree_build(X_right, right_y, right_n, n_features, n_classes,
                              depth + 1, max_depth, min_samples_split,
                              min_samples_leaf, max_features);

    // 释放临时数组，X_left/X_right 只释放指针数组本身，不释放原始数据行
    free(X_left);
    free(X_right);
    free(indices);
    free(feature_pool);
    free(left_idx);
    free(right_idx);
    free(left_y);
    free(right_y);

    return node;
}

int tree_predict(TreeNode *node, double *sample) {
    if (!node) return -1;
    // label >= 0 表示已经到达叶子节点，直接返回类别
    if (node->label >= 0) return node->label;
    // 内部节点根据 feature/threshold 判断样本进入左子树还是右子树
    if (sample[node->feature] <= node->threshold) {
        return tree_predict(node->left, sample);
    } else {
        return tree_predict(node->right, sample);
    }
}

void tree_free(TreeNode *node) {
    if (!node) return;
    // 后序释放：先释放左右子树，再释放当前节点
    tree_free(node->left);
    tree_free(node->right);
    free(node);
}

// 创建随机森林对象，只分配森林结构和树指针数组，具体决策树在 rf_train 中生成
RandomForest *rf_create(int n_trees, int max_depth, int n_features, int n_classes,
                         int min_samples_split, int min_samples_leaf) {
    // 随机森林常用设置：每次分裂随机选择 sqrt(n_features) 个特征
    int max_features = (int)sqrt((double)n_features);
    if (max_features < 1) max_features = 1;

    RandomForest *rf = (RandomForest *)malloc(sizeof(RandomForest));
    rf->n_trees = n_trees;
    rf->max_depth = max_depth;
    rf->n_features = n_features;
    rf->n_classes = n_classes;
    rf->min_samples_split = min_samples_split;
    rf->min_samples_leaf = min_samples_leaf;
    rf->max_features = max_features;
    rf->trees = (TreeNode **)calloc((size_t)n_trees, sizeof(TreeNode *));
    return rf;
}

// 训练随机森林：每棵树使用 Bootstrap 样本训练，并用 OOB 样本估计泛化性能
void rf_train(RandomForest *rf, double **X, int *y, int n_samples) {
    int t, i;

    // bs 是单棵树的 Bootstrap 训练集，反复复用以减少重复分配
    Dataset *bs = (Dataset *)malloc(sizeof(Dataset));
    bs->n_samples = n_samples;
    bs->n_features = rf->n_features;
    bs->X = (double **)malloc((size_t)n_samples * sizeof(double *));
    bs->y = (int *)malloc((size_t)n_samples * sizeof(int));
    for (i = 0; i < n_samples; i++) {
        bs->X[i] = (double *)malloc((size_t)rf->n_features * sizeof(double));
    }

    // OOB 累积投票：oob_votes[i][c] 表示把样本 i 预测为类别 c 的树的数量
    int **oob_votes = (int **)malloc((size_t)n_samples * sizeof(int *));
    int *oob_count = (int *)calloc((size_t)n_samples, sizeof(int));
    for (i = 0; i < n_samples; i++) {
        oob_votes[i] = (int *)calloc((size_t)rf->n_classes, sizeof(int));
    }

    printf("Training Random Forest with %d trees...\n", rf->n_trees);
    // 逐棵训练决策树
    for (t = 0; t < rf->n_trees; t++) {
        // Bootstrap 有放回采样，并记录哪些原始样本被抽入袋内
        int *in_bag = (int *)calloc((size_t)n_samples, sizeof(int));
        for (i = 0; i < n_samples; i++) {
            int idx = rand() % n_samples;
            in_bag[idx] = 1;
            memcpy(bs->X[i], X[idx], (size_t)rf->n_features * sizeof(double));
            bs->y[i] = y[idx];
        }

        // 用当前 Bootstrap 样本训练一棵 CART 决策树
        rf->trees[t] = tree_build(bs->X, bs->y, n_samples,
                                   rf->n_features, rf->n_classes,
                                   0, rf->max_depth,
                                   rf->min_samples_split,
                                   rf->min_samples_leaf,
                                   rf->max_features);

        // OOB 评估：用当前树预测所有没有被抽入袋内的样本
        for (i = 0; i < n_samples; i++) {
            if (!in_bag[i]) {
                int pred = tree_predict(rf->trees[t], X[i]);
                if (pred >= 0 && pred < rf->n_classes) oob_votes[i][pred]++;
                oob_count[i]++;
            }
        }
        free(in_bag);

        if ((t + 1) % 10 == 0 || t == rf->n_trees - 1) {
            // 每训练若干棵树，就用已累积的 OOB 投票估计当前准确率
            int oob_correct = 0, oob_total = 0;
            for (i = 0; i < n_samples; i++) {
                if (oob_count[i] > 0) {
                    oob_total++;
                    if (argmax(oob_votes[i], rf->n_classes) == y[i]) oob_correct++;
                }
            }
            double oob_acc = oob_total > 0
                ? 100.0 * oob_correct / oob_total : 0.0;
            printf("  Tree %d/%d | OOB accuracy: %.2f%% (%d/%d)\n",
                   t + 1, rf->n_trees, oob_acc, oob_correct, oob_total);
        }
    }

    // 释放 Bootstrap 数据集和 OOB 投票数组
    for (i = 0; i < n_samples; i++) {
        free(bs->X[i]);
        free(oob_votes[i]);
    }
    free(bs->X);
    free(bs->y);
    free(bs);
    free(oob_votes);
    free(oob_count);
}

int rf_predict(RandomForest *rf, double *sample) {
    // 硬投票：每棵树给出一个类别，最终选择得票最多的类别
    int *votes = (int *)calloc((size_t)rf->n_classes, sizeof(int));
    int t;
    for (t = 0; t < rf->n_trees; t++) {
        int pred = tree_predict(rf->trees[t], sample);
        if (pred >= 0 && pred < rf->n_classes) votes[pred]++;
    }
    int result = argmax(votes, rf->n_classes);
    free(votes);
    return result;
}

// 软投票：返回各类别的得票比例，反映预测的不确定性
double *rf_predict_proba(RandomForest *rf, double *sample) {
    double *probs = (double *)calloc((size_t)rf->n_classes, sizeof(double));
    int t, valid = 0;
    for (t = 0; t < rf->n_trees; t++) {
        int pred = tree_predict(rf->trees[t], sample);
        if (pred >= 0 && pred < rf->n_classes) {
            probs[pred] += 1.0;
            valid++;
        }
    }
    if (valid > 0) {
        for (t = 0; t < rf->n_classes; t++) probs[t] /= (double)valid;
    }
    return probs;
}

void rf_free(RandomForest *rf) {
    if (!rf) return;
    // 依次释放所有决策树，再释放树数组和森林结构
    int t;
    for (t = 0; t < rf->n_trees; t++) tree_free(rf->trees[t]);
    free(rf->trees);
    free(rf);
}

// ========== 模型序列化 ==========
// 文件格式：
// 第 1 行：树的数量
// 第 2 行：最大深度、特征数、类别数、最小分裂样本数、最小叶子样本数
// 后续内容：每棵树的前序遍历结果，T 表示节点，N 表示空节点

// 保存单个节点和它的左右子树
static void node_save(TreeNode *node, FILE *fp) {
    if (!node) {
        fprintf(fp, "N\n");
        return;
    }
    // T 行保存一个节点，随后递归保存左右子树；N 表示空子节点
    fprintf(fp, "T %d %.8f %d\n", node->feature, node->threshold, node->label);
    node_save(node->left, fp);
    node_save(node->right, fp);
}

// 从文件中递归读取单个节点和它的左右子树
static TreeNode *node_load(FILE *fp) {
    char type;
    if (fscanf(fp, " %c", &type) != 1) return NULL;
    if (type == 'N') return NULL;

    // 按 node_save 的前序格式递归恢复树结构
    TreeNode *node = (TreeNode *)calloc(1, sizeof(TreeNode));
    if (fscanf(fp, "%d %lf %d", &node->feature, &node->threshold, &node->label) != 3) {
        free(node);
        return NULL;
    }
    node->left = node_load(fp);
    node->right = node_load(fp);
    return node;
}

int rf_save(RandomForest *rf, const char *filename) {
    // 打开模型文件用于写入
    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;

    // 写入随机森林整体参数
    fprintf(fp, "%d\n", rf->n_trees);
    fprintf(fp, "%d %d %d %d %d\n",
            rf->max_depth, rf->n_features, rf->n_classes,
            rf->min_samples_split, rf->min_samples_leaf);

    // 逐棵保存决策树
    int t;
    for (t = 0; t < rf->n_trees; t++) node_save(rf->trees[t], fp);
    fclose(fp);
    return 0;
}

RandomForest *rf_load(const char *filename) {
    // 打开模型文件用于读取
    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;

    // 读取随机森林整体参数
    int n_trees, max_depth, n_features, n_classes, min_samples_split, min_samples_leaf;
    if (fscanf(fp, "%d", &n_trees) != 1) { fclose(fp); return NULL; }
    if (fscanf(fp, "%d %d %d %d %d",
               &max_depth, &n_features, &n_classes,
               &min_samples_split, &min_samples_leaf) != 5) {
        fclose(fp);
        return NULL;
    }

    // 按读取到的参数创建随机森林，并逐棵恢复决策树
    RandomForest *rf = rf_create(n_trees, max_depth, n_features, n_classes,
                                  min_samples_split, min_samples_leaf);
    int t;
    for (t = 0; t < n_trees; t++) rf->trees[t] = node_load(fp);
    fclose(fp);
    return rf;
}
