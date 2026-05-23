# 编译器和编译选项
CC = gcc
CFLAGS = -Wall -O2 -lm
TARGET = robot_rf
# 默认目标：编译
all: $(TARGET)
$(TARGET): robot_rf.c
	$(CC) $^ -o $@ $(CFLAGS)
# 运行程序
run: $(TARGET)
	./$(TARGET)
# 清理编译产物
clean:
	rm -f $(TARGET) *.o *.exe
# 调试模式编译（带调试信息）
debug:
	$(CC) robot_rf.c -o $(TARGET)_debug -g -Wall -lm