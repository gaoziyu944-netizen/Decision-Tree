# 编译器和编译选项
CC = gcc
CFLAGS = -Wall -O2 -Iinclude -I. -lm
TARGET = robot_rf

# 源文件
SRCS = src/main.c src/robot_rf.c src/utils.c

# 默认目标：编译
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(SRCS) -o $@ $(CFLAGS)

# 运行程序
run: $(TARGET)
	./$(TARGET)

# 训练模式
train: $(TARGET)
	./$(TARGET) train data/train.csv models/robot_model.bin

# 预测模式
predict: $(TARGET)
	./$(TARGET) predict models/robot_model.bin data/test.csv output.csv

# 评估模式
evaluate: $(TARGET)
	./$(TARGET) evaluate models/robot_model.bin data/test.csv

# # 清理编译产物
# clean:
# 	rm -f $(TARGET) *.o *.exe output.csv
