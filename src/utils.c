// utils.c - 工具函数实现
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
// 工具函数和 Dataset 结构体声明
#include "utils.h"

// 随机数播种标记：保证 srand 只调用一次
static int g_seeded = 0;

// 惰性播种：第一次调用随机函数时才使用当前时间播种
static void ensure_seed(void) {
    if (!g_seeded) {
        srand((unsigned int)time(NULL));
        g_seeded = 1;
    }
}

double rand_double(void) {
    // C 标准库 rand() 返回整数，这里归一化到 [0, 1] 区间
    ensure_seed();
    return (double)rand() / (double)RAND_MAX;
}

int rand_int(int max) {
    // max 非法时返回 0，避免 rand() % 0
    ensure_seed();
    if (max <= 0) return 0;
    return rand() % max;
}

// Fisher-Yates 洗牌：从末尾开始，每个位置与 [0, i] 内随机位置交换
void shuffle_indices(int *indices, int n) {
    int i, j, tmp;
    ensure_seed();
    for (i = n - 1; i > 0; i--) {
        j = rand() % (i + 1);
        tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

void shuffle_features(int *features, int n) {
    // 与 shuffle_indices 使用同样的洗牌算法，用于随机选择候选特征
    int i, j, tmp;
    ensure_seed();
    for (i = n - 1; i > 0; i--) {
        j = rand() % (i + 1);
        tmp = features[i];
        features[i] = features[j];
        features[j] = tmp;
    }
}

int argmax(int *vals, int n) {
    // 顺序扫描最大值；如果有并列，保留最先出现的类别
    int i, best = 0, best_val = vals[0];
    for (i = 1; i < n; i++) {
        if (vals[i] > best_val) {
            best_val = vals[i];
            best = i;
        }
    }
    return best;
}

// 计算 Gini 不纯度：1 - Σ(p_c²)，纯节点为 0，类别越混杂值越大
double gini_impurity(int *y, int n_samples, int n_classes) {
    int *counts;
    double gini = 1.0;
    int i;

    if (n_samples == 0) return 0.0;

    counts = (int *)calloc((size_t)n_classes, sizeof(int));
    for (i = 0; i < n_samples; i++) counts[y[i]]++;
    for (i = 0; i < n_classes; i++) {
        double p = (double)counts[i] / (double)n_samples;
        gini -= p * p;
    }
    free(counts);
    return gini;
}

static int count_lines(FILE *fp) {
    // 统计文件行数，调用结束后恢复原来的文件位置
    int count = 0;
    char ch;
    long pos = ftell(fp);
    rewind(fp);
    while ((ch = (char)fgetc(fp)) != EOF) {
        if (ch == '\n') count++;
    }
    // 恢复文件指针，避免影响后续读取
    fseek(fp, pos, SEEK_SET);
    return count;
}

static int count_fields(const char *line) {
    // CSV 字段数 = 逗号数量 + 1，这里假设没有带引号的复杂字段
    int count = 0;
    const char *p = line;
    while (*p) { if (*p == ',') count++; p++; }
    return count + 1;
}

// 从 CSV 文件加载数据集
// 第 1 行是表头，用来推断特征数量；后续每行格式为：特征1,特征2,...,标签
Dataset *dataset_load(const char *filename) {
    FILE *fp;
    Dataset *ds;
    char line[4096];
    int n_samples, n_features;
    int i, j;

    // 打开 CSV 文件
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        return NULL;
    }

    // 统计样本行数
    n_samples = count_lines(fp);
    if (n_samples == 0) {
        fprintf(stderr, "Error: empty file %s\n", filename);
        fclose(fp);
        return NULL;
    }

    // 读取表头，并根据字段数推断特征数量
    fseek(fp, 0, SEEK_SET);
    if (fgets(line, sizeof(line), fp) == NULL) {
        fprintf(stderr, "Error: cannot read header\n");
        fclose(fp);
        return NULL;
    }
    // 最后一列是标签，所以特征数 = 字段数 - 1
    n_features = count_fields(line) - 1;

    // 分配 Dataset 结构体和数组
    ds = (Dataset *)malloc(sizeof(Dataset));
    ds->n_samples = n_samples;
    ds->n_features = n_features;

    ds->X = (double **)malloc((size_t)n_samples * sizeof(double *));
    ds->y = (int *)malloc((size_t)n_samples * sizeof(int));

    // 逐行读取数据，每行前 n_features 列作为特征，最后一列作为标签
    for (i = 0; i < n_samples; i++) {
        if (fgets(line, sizeof(line), fp) == NULL) {
            ds->n_samples = i;
            break;
        }
        ds->X[i] = (double *)malloc((size_t)n_features * sizeof(double));
        // strtok 按逗号切分当前行
        char *token = strtok(line, ",");
        for (j = 0; j < n_features && token != NULL; j++) {
            ds->X[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
        ds->y[i] = (token != NULL) ? atoi(token) : 0;
    }
    fclose(fp);
    return ds;
}

void dataset_free(Dataset *ds) {
    int i;
    if (!ds) return;
    // 先释放每一行特征，再释放行指针数组和标签数组
    for (i = 0; i < ds->n_samples; i++) free(ds->X[i]);
    free(ds->X);
    free(ds->y);
    free(ds);
}

// Bootstrap 有放回抽样
// 每次从原始数据集中随机抽取一条样本复制到目标数据集
void dataset_bootstrap(Dataset *src, Dataset *dst) {
    int i, idx;
    ensure_seed();
    for (i = 0; i < src->n_samples; i++) {
        // 随机选择一个原始样本
        idx = rand() % src->n_samples;
        memcpy(dst->X[i], src->X[idx], (size_t)src->n_features * sizeof(double));
        dst->y[i] = src->y[idx];
    }
}
