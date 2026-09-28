from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO('yolov8s-cls.pt')

    model.train(
        data=r'D:\test data\dataset',   # test 和 data 中间有空格
        epochs=100,                      # 先测 10 轮
        imgsz=224,
        batch=16,                       # 8GB 显存，16 够稳，想快点改 32
        device=0,                       # 用 RTX 5050
        workers=4,                      # 报 DataLoader 错就改成 0
        amp=True,                       # 混合精度提速
        cache='ram'                     # 数据缓存到内存，提速
    )