def exponential_moving_average(arr, alpha=0.5):
    ema = arr[0]  # 初始 EMA 設為第一個元素
    for i in range(1, len(arr)):
        ema = alpha * arr[i] + (1 - alpha) * ema
    return ema

# 測試
arr = [10, 12, 14, 16, 18]
print("Exponential Moving Average:", exponential_moving_average(arr))
