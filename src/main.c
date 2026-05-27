// main.c - 命令行入口，支持 train / predict / input / single / evaluate / features 六种模式
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// 工具函数头文件：负责数据集读取、随机数、统计函数
#include "utils.h"
// 随机森林核心头文件：负责模型训练、预测、保存和加载
#include "robot_rf.h"
// 项目配置头文件：保存树深度、树数量、类别数等参数
#include "config.h"

// 机器人动作编号到动作名称的映射
static const char *action_names[] = {"FORWARD", "LEFT", "RIGHT", "STOP"};

// 打印命令行帮助信息，用户参数不足或模式错误时调用
static void print_usage(void) {
    printf("Robot Decision Forest - Random Forest for Robot Control\n");
    printf("========================================================\n\n");
    printf("Usage:\n");
    printf("  robot_rf train <data.csv> <model_output.bin>\n");
    printf("  robot_rf predict <model.bin> <data.csv> [output.csv]\n");
    printf("  robot_rf input <model.bin>\n");
    printf("  robot_rf single <model.bin> <dist_front> <dist_left> <dist_right> <speed>\n");
    printf("  robot_rf evaluate <model.bin> <test.csv>\n");
    printf("  robot_rf features <model.bin>\n\n");
    printf("Modes:\n");
    printf("  train    - Train model on CSV data and save to file\n");
    printf("  predict  - Load model and predict actions from CSV data\n");
    printf("  input    - Type sensor values in the console and predict one action\n");
    printf("  single   - Predict one action from command-line sensor values\n");
    printf("  evaluate - Evaluate model on labeled test data with metrics\n");
    printf("  features - Display model feature importance\n\n");
    printf("Config (config.h):\n");
    printf("  MAX_DEPTH         = %d\n", MAX_DEPTH);
    printf("  N_TREES           = %d\n", N_TREES);
    printf("  MIN_SAMPLES_SPLIT = %d\n", MIN_SAMPLES_SPLIT);
    printf("  MIN_SAMPLES_LEAF  = %d\n", MIN_SAMPLES_LEAF);
    printf("  N_FEATURES        = %d\n", N_FEATURES);
    printf("  N_CLASSES         = %d\n", N_CLASSES);
}

static int do_train(const char *data_file, const char *model_file) {
    Dataset *ds;
    RandomForest *rf;

    // 训练流程：读取 CSV -> 创建随机森林 -> 训练模型 -> 保存模型
    printf("Loading training data from: %s\n", data_file);
    ds = dataset_load(data_file);
    if (!ds) return 1;

    printf("Loaded %d samples, %d features\n", ds->n_samples, ds->n_features);

    // 根据配置参数创建随机森林对象
    rf = rf_create(N_TREES, MAX_DEPTH, ds->n_features, N_CLASSES,
                   MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF);
    // 使用训练集特征 X 和标签 y 训练随机森林
    rf_train(rf, ds->X, ds->y, ds->n_samples);

    // 将训练好的模型保存到文件，供 predict/evaluate/features 模式加载
    printf("\nSaving model to: %s\n", model_file);
    if (rf_save(rf, model_file) != 0) {
        fprintf(stderr, "Error: failed to save model\n");
        rf_free(rf);
        dataset_free(ds);
        return 1;
    }
    printf("Model saved successfully\n");

    // 计算训练集准确率，仅用于参考，真实泛化能力应看 OOB 或测试集
    printf("\nFinal evaluation on training data:\n");
    {
        int correct = 0, i;
        for (i = 0; i < ds->n_samples; i++) {
            int pred = rf_predict(rf, ds->X[i]);
            if (pred == ds->y[i]) correct++;
        }
        printf("  Training accuracy: %.2f%% (%d/%d)\n",
               100.0 * correct / ds->n_samples, correct, ds->n_samples);
    }

    // 释放模型和数据集内存
    rf_free(rf);
    dataset_free(ds);
    return 0;
}

static int do_predict(const char *model_file, const char *data_file,
                       const char *output_file) {
    RandomForest *rf;
    Dataset *ds;
    FILE *out;
    int i;

    // 预测流程：加载已训练模型，再逐行读取样本并输出类别和概率
    printf("Loading model from: %s\n", model_file);
    rf = rf_load(model_file);
    if (!rf) {
        fprintf(stderr, "Error: cannot load model %s\n", model_file);
        return 1;
    }
    printf("Model loaded: %d trees, %d features, %d classes\n",
           rf->n_trees, rf->n_features, rf->n_classes);

    printf("Loading data from: %s\n", data_file);
    ds = dataset_load(data_file);
    if (!ds) {
        rf_free(rf);
        return 1;
    }

    // 如果命令行没有指定输出文件，则使用默认文件名
    if (!output_file) output_file = "output.csv";

    // 打开预测结果输出文件
    out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: cannot open output file %s\n", output_file);
        dataset_free(ds);
        rf_free(rf);
        return 1;
    }

    printf("Predicting... Results saved to: %s\n\n", output_file);
    // 写入 CSV 表头，包含预测类别和每个动作的概率
    fprintf(out, "predicted_action,action_name");
    for (i = 0; i < rf->n_classes; i++) {
        fprintf(out, ",prob_%s", action_names[i]);
    }
    fprintf(out, "\n");

    // 遍历所有样本，逐条预测
    for (i = 0; i < ds->n_samples; i++) {
        int pred = rf_predict(rf, ds->X[i]);
        double *probs = rf_predict_proba(rf, ds->X[i]);
        const char *name = (pred >= 0 && pred < 4) ? action_names[pred] : "UNKNOWN";

        // 写入预测结果和每个类别的概率
        fprintf(out, "%d,%s", pred, name);
        {
            int c;
            for (c = 0; c < rf->n_classes; c++) {
                fprintf(out, ",%.4f", probs[c]);
            }
        }
        fprintf(out, "\n");

        // 在控制台打印当前样本的预测结果
        printf("  Sample %d: %s (%d)", i + 1, name, pred);
        if (ds->y) {
            const char *true_name = (ds->y[i] >= 0 && ds->y[i] < 4)
                                     ? action_names[ds->y[i]] : "UNKNOWN";
            printf(" | True: %s (%d)", true_name, ds->y[i]);
            if (pred == ds->y[i]) printf(" ✓");
        }
        printf("\n");
        // rf_predict_proba 动态分配概率数组，使用完必须释放
        free(probs);
    }

    // 关闭文件并释放内存
    fclose(out);
    dataset_free(ds);
    rf_free(rf);
    return 0;
}

static int print_single_prediction(RandomForest *rf, double sample[N_FEATURES]) {
    int c;
    // 对单条传感器样本进行预测，pred 是最终投票得到的动作类别
    int pred = rf_predict(rf, sample);
    // 获取每个动作类别的预测概率，便于观察模型不确定性
    double *probs = rf_predict_proba(rf, sample);
    const char *name = (pred >= 0 && pred < N_CLASSES) ? action_names[pred] : "UNKNOWN";

    // 打印最终预测动作
    printf("\nPredicted action: %s (%d)\n", name, pred);
    printf("Probabilities:\n");
    // 逐类打印概率，概率由随机森林中投票给该类的树占比得到
    for (c = 0; c < rf->n_classes; c++) {
        const char *class_name = (c >= 0 && c < N_CLASSES) ? action_names[c] : "UNKNOWN";
        printf("  %s: %.2f%%\n", class_name, probs[c] * 100.0);
    }

    // rf_predict_proba 返回动态分配的数组，打印完成后必须释放
    free(probs);
    return pred;
}

static int do_input(const char *model_file) {
    RandomForest *rf;
    double sample[N_FEATURES];

    // 交互式预测流程：加载模型 -> 从控制台读取 4 个传感器值 -> 输出预测动作
    printf("Loading model from: %s\n", model_file);
    rf = rf_load(model_file);
    if (!rf) {
        fprintf(stderr, "Error: cannot load model %s\n", model_file);
        return 1;
    }

    printf("Enter sensor values:\n");
    // 依次读取前方距离、左侧距离、右侧距离和速度
    printf("  dist_front: ");
    if (scanf("%lf", &sample[0]) != 1) goto input_error;
    printf("  dist_left: ");
    if (scanf("%lf", &sample[1]) != 1) goto input_error;
    printf("  dist_right: ");
    if (scanf("%lf", &sample[2]) != 1) goto input_error;
    printf("  speed: ");
    if (scanf("%lf", &sample[3]) != 1) goto input_error;

    // 复用单样本预测输出函数
    print_single_prediction(rf, sample);
    rf_free(rf);
    return 0;

input_error:
    // 任意一个输入不是合法数字时，跳转到这里统一处理错误和释放内存
    fprintf(stderr, "Error: invalid sensor value\n");
    rf_free(rf);
    return 1;
}

static int do_single(const char *model_file, char *argv[]) {
    RandomForest *rf;
    double sample[N_FEATURES];
    int i;

    // 命令行单样本预测：从 argv 中解析 4 个传感器数值
    for (i = 0; i < N_FEATURES; i++) {
        char *endptr = NULL;
        sample[i] = strtod(argv[i], &endptr);
        // endptr 用来检查整个字符串是否都被成功解析为数字
        if (endptr == argv[i] || *endptr != '\0') {
            fprintf(stderr, "Error: invalid sensor value: %s\n", argv[i]);
            return 1;
        }
    }

    // 加载模型后，对解析出的单条样本执行预测
    printf("Loading model from: %s\n", model_file);
    rf = rf_load(model_file);
    if (!rf) {
        fprintf(stderr, "Error: cannot load model %s\n", model_file);
        return 1;
    }

    print_single_prediction(rf, sample);
    rf_free(rf);
    return 0;
}

static int do_evaluate(const char *model_file, const char *data_file) {
    RandomForest *rf;
    Dataset *ds;
    int i, c;
    // 混淆矩阵：[真实类别][预测类别]
    int confusion[16][16] = {{0}};
    int correct = 0;

    // 评估流程：加载模型和带标签测试集，统计准确率、混淆矩阵和分类指标
    printf("Loading model from: %s\n", model_file);
    rf = rf_load(model_file);
    if (!rf) {
        fprintf(stderr, "Error: cannot load model %s\n", model_file);
        return 1;
    }
    printf("Model loaded: %d trees, %d features, %d classes\n",
           rf->n_trees, rf->n_features, rf->n_classes);

    printf("Loading test data from: %s\n", data_file);
    ds = dataset_load(data_file);
    if (!ds) {
        rf_free(rf);
        return 1;
    }
    printf("Loaded %d samples\n", ds->n_samples);

    // 遍历测试集，填充混淆矩阵并统计预测正确的样本数
    for (i = 0; i < ds->n_samples; i++) {
        int pred = rf_predict(rf, ds->X[i]);
        int actual = ds->y[i];
        // confusion[真实类别][预测类别] 记录每种真实/预测组合的次数
        if (pred >= 0 && pred < ds->n_samples && actual >= 0 && actual < ds->n_samples) {
            confusion[actual][pred]++;
            if (pred == actual) correct++;
        }
    }

    // 计算整体准确率
    double accuracy = 100.0 * correct / ds->n_samples;
    printf("\nEvaluation Results\n");
    printf("==================\n");
    printf("Overall accuracy: %.2f%% (%d/%d)\n\n", accuracy, correct, ds->n_samples);

    // 打印混淆矩阵
    printf("Confusion Matrix:\n");
    printf("                ");
    for (c = 0; c < rf->n_classes; c++) {
        printf("Pred(%s)  ", action_names[c]);
    }
    printf("\n");
    for (c = 0; c < rf->n_classes; c++) {
        printf("  True(%s)  ", action_names[c]);
        {
            int c2;
            for (c2 = 0; c2 < rf->n_classes; c2++) {
                printf("    %4d  ", confusion[c][c2]);
            }
        }
        printf("\n");
    }

    // 计算每个类别的 Precision、Recall 和 F1 分数
    printf("\nPer-Class Metrics:\n");
    printf("%-10s  %8s  %8s  %8s  %8s\n",
           "Class", "Prec(%)", "Recall(%)", "F1(%)", "Support");
    for (c = 0; c < rf->n_classes; c++) {
        // 真正例：混淆矩阵对角线表示预测正确的数量
        int tp = confusion[c][c];
        int pred_total = 0, true_total = 0;
        int c2;
        for (c2 = 0; c2 < rf->n_classes; c2++) {
            // 行和表示真实属于该类的样本数
            true_total += confusion[c][c2];
            // 列和表示被预测为该类的样本数
            pred_total += confusion[c2][c];
        }
        double prec = pred_total > 0 ? 100.0 * tp / pred_total : 0.0;
        double rec  = true_total > 0 ? 100.0 * tp / true_total : 0.0;
        double f1   = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
        printf("%-10s  %8.2f  %8.2f  %8.2f  %8d\n",
               action_names[c], prec, rec, f1, true_total);
    }

    // 释放内存
    dataset_free(ds);
    rf_free(rf);
    return 0;
}

static int do_features(const char *model_file) {
    RandomForest *rf;
    int *feature_counts, t, i;
    // 传感器特征名称
    const char *feature_names[] = {"dist_front", "dist_left", "dist_right", "speed"};

    // 特征分析：统计所有树中每个特征被用作分裂条件的次数
    printf("Loading model from: %s\n", model_file);
    rf = rf_load(model_file);
    if (!rf) {
        fprintf(stderr, "Error: cannot load model %s\n", model_file);
        return 1;
    }

    // 初始化特征分裂次数计数数组
    feature_counts = (int *)calloc((size_t)rf->n_features, sizeof(int));

    printf("\nFeature Analysis (%d trees):\n", rf->n_trees);
    printf("============================\n");

    // 遍历随机森林中的每一棵决策树
    for (t = 0; t < rf->n_trees; t++) {
        if (!rf->trees[t]) continue;

        // 使用显式栈进行 DFS 遍历，避免递归过深导致栈溢出
        int tree_depth = 0;
        TreeNode *stack[256];
        int depths[256];
        int top = 0;

        stack[0] = rf->trees[t];
        depths[0] = 0;

        while (top >= 0) {
            TreeNode *n = stack[top];
            int d = depths[top];
            top--;

            // 空节点或叶子节点不参与特征分裂统计
            if (!n || n->label >= 0) {
                if (d > tree_depth) tree_depth = d;
                continue;
            }

            // 当前内部节点使用了 n->feature 作为分裂特征
            feature_counts[n->feature]++;

            // 右子树入栈
            top++;
            if (top >= 256) break;
            stack[top] = n->right;
            depths[top] = d + 1;

            // 左子树入栈
            top++;
            if (top >= 256) break;
            stack[top] = n->left;
            depths[top] = d + 1;
        }
    }

    // 统计所有特征的总分裂次数
    int total_splits = 0;
    for (i = 0; i < rf->n_features; i++) {
        total_splits += feature_counts[i];
    }

    // 特征重要性 = 该特征分裂次数 / 总分裂次数
    for (i = 0; i < rf->n_features; i++) {
        double importance = total_splits > 0
            ? 100.0 * feature_counts[i] / total_splits : 0.0;
        printf("  %s: %d splits (%.1f%% importance)\n",
               feature_names[i], feature_counts[i], importance);
    }

    // 释放内存
    free(feature_counts);
    rf_free(rf);
    return 0;
}

int main(int argc, char *argv[]) {
    // 参数不足时打印帮助信息
    if (argc < 2) {
        print_usage();
        return 0;
    }

    // 根据第一个命令行参数分发到对应功能模式
    if (strcmp(argv[1], "train") == 0) {
        if (argc < 4) {
            printf("Usage: robot_rf train <data.csv> <model_output.bin>\n");
            return 1;
        }
        return do_train(argv[2], argv[3]);
    }
    else if (strcmp(argv[1], "predict") == 0) {
        if (argc < 4) {
            printf("Usage: robot_rf predict <model.bin> <data.csv> [output.csv]\n");
            return 1;
        }
        return do_predict(argv[2], argv[3], argc > 4 ? argv[4] : NULL);
    }
    else if (strcmp(argv[1], "input") == 0) {
        if (argc < 3) {
            printf("Usage: robot_rf input <model.bin>\n");
            return 1;
        }
        return do_input(argv[2]);
    }
    else if (strcmp(argv[1], "single") == 0) {
        if (argc < 7) {
            printf("Usage: robot_rf single <model.bin> <dist_front> <dist_left> <dist_right> <speed>\n");
            return 1;
        }
        return do_single(argv[2], &argv[3]);
    }
    else if (strcmp(argv[1], "evaluate") == 0) {
        if (argc < 4) {
            printf("Usage: robot_rf evaluate <model.bin> <test.csv>\n");
            return 1;
        }
        return do_evaluate(argv[2], argv[3]);
    }
    else if (strcmp(argv[1], "features") == 0) {
        if (argc < 3) {
            printf("Usage: robot_rf features <model.bin>\n");
            return 1;
        }
        return do_features(argv[2]);
    }
    else {
        // 未知模式时打印帮助信息
        print_usage();
        return 0;
    }
}
