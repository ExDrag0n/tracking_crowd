import cv2
import numpy as np


def load_yolo():
    """
    Загрузка модели и весов YOLOv4.

    Returns:
        net (cv2.dnn_DetectionModel): Загруженная модель YOLOv4
        classes (list): список имён объектов класса
    """
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes


def detect_objects(net, frame, classes):
    """
    Обнаружение объектов в фрейме с помощью YOLOv4.

    Args:
        net (cv2.dnn_DetectionModel): Загруженная модель YOLOv4
        frame (numpy.ndarray): Обрабатываемый кадр
        classes (list): список имён объектов класса

    Returns:
        outputs (list): Список найденных объектов
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, class_ids, indexes


def draw_bounding_boxes(frame, boxes, confidences, class_ids, indexes, classes):
    """
    Обрисовка найденных объектов.

    Args:
        frame (numpy.ndarray): Обрабатываемый кадр
        boxes (list): Список координат контура обрисовки
        confidences (list): Список значений уверенности
        class_ids (list): Список ID классов
        indexes (list): Список допустимых индексов объектов после NMS
        classes (list): список имён объектов класса
    """
    color = (255, 0, 0)  # Красный цвет для всех рамок
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = round(confidences[i], 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence}", (x, y + 30), font, 1, color, 2)



def main():
    """
    Основная функция для запуска отслеживания и вывода результата.
    """
    net, classes = load_yolo()

    cap = cv2.VideoCapture('crowd.mp4')
    if not cap.isOpened():
        print("Ошибка при открытии файла")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids, indexes = detect_objects(net, frame, classes)
        draw_bounding_boxes(frame, boxes, confidences, class_ids, indexes, classes)

        out.write(frame)

        frame_count += 1

    cap.release()
    out.release()

    print(f"Обработано {frame_count} кадров.")


if __name__ == "__main__":
    main()
