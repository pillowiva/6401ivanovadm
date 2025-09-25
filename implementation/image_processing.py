import numpy as np
import math


class ImageProcessing():
    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        '''
        Свертка
        1. определяем размер изображения и ядра
        2. определяем необходимые отступы для обработки краев (чтобы ядро могло пройти по всему исходному изображению)
        3. добавляем к исходному изображения необходимые отступы
        4. создаем выходное изображение
        5. для каждого пикселя вычисляем взвешенную сумму с ядром
        '''

        img_height, img_width = image.shape
        kernel_height, kernel_width = kernel.shape

        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        output = np.zeros_like(image, dtype=np.float32)

        for i in range(img_height):
            for j in range(img_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                output[i, j] = np.sum(region * kernel)

        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        '''
        Преобразование в ЧБ
        1. проверяем, является ли изображения цветным и количество каналов у изображения - 3 (изображения с альфа-каналом не рассматриваем)
        2. берем срез всех элементов по первым трем каналам и производим матричное умножение на весовые коэффициенты
        '''
        if len(image.shape) == 3 and image.shape[2] == 3:
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            return image

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma)
        return (corrected * 255).astype(np.uint8)

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray_image = self._rgb_to_grayscale(image)
        else:
            gray_image = image

        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16
        smoothed = self._convolution(gray_image, gaussian_kernel)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        grad_x = self._convolution(smoothed, sobel_x)
        grad_y = self._convolution(smoothed, sobel_y)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return magnitude.astype(np.uint8)


    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray_image = self._rgb_to_grayscale(image)
        else:
            gray_image = image

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        Ix = self._convolution(gray_image.astype(np.float32), sobel_x)
        Iy = self._convolution(gray_image.astype(np.float32), sobel_y)

        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy

        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16

        Sx2 = self._convolution(Ix2, gaussian_kernel)
        Sy2 = self._convolution(Iy2, gaussian_kernel)
        Sxy = self._convolution(Ixy, gaussian_kernel)

        det = Sx2 * Sy2 - Sxy * Sxy
        trace = Sx2 + Sy2

        k = 0.04
        R = det - k * trace * trace

        threshold = 0.01 * R.max()
        corners = R > threshold

        if len(image.shape) == 3:
            result = image.copy()
            result[corners] = [255, 0, 0]
        else:
            result = np.stack([image] * 3, axis=-1)
            result[corners] = [255, 0, 0]

        return result.astype(np.uint8)

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray_image = self._rgb_to_grayscale(image)
        else:
            gray_image = image

        edges = self.edge_detection(gray_image)

        height, width = edges.shape

        min_radius = 10
        max_radius = min(height, width) // 4

        accumulator = np.zeros((height, width, max_radius - min_radius + 1))

        edge_points = np.argwhere(edges > 0)

        for y, x in edge_points:
            for r in range(min_radius, max_radius + 1):
                for theta in range(0, 360, 5):
                    a = int(x - r * math.cos(math.radians(theta)))
                    b = int(y - r * math.sin(math.radians(theta)))

                    if 0 <= a < width and 0 <= b < height:
                        accumulator[b, a, r - min_radius] += 1

        circles = []
        threshold = 0.8 * accumulator.max()

        for r_idx in range(accumulator.shape[2]):
            radius = r_idx + min_radius
            for y in range(accumulator.shape[0]):
                for x in range(accumulator.shape[1]):
                    if accumulator[y, x, r_idx] > threshold:
                        is_local_max = True
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                for dr in range(-1, 2):
                                    ny, nx, nr = y + dy, x + dx, r_idx + dr
                                    if (0 <= ny < accumulator.shape[0] and
                                            0 <= nx < accumulator.shape[1] and
                                            0 <= nr < accumulator.shape[2]):
                                        if accumulator[ny, nx, nr] > accumulator[y, x, r_idx]:
                                            is_local_max = False
                                            break

                        if is_local_max:
                            circles.append((x, y, radius))

        if len(image.shape) == 3:
            result = image.copy()
        else:
            result = np.stack([image] * 3, axis=-1)

        for x, y, r in circles:
            for angle in range(0, 360, 5):
                px = int(x + r * math.cos(math.radians(angle)))
                py = int(y + r * math.sin(math.radians(angle)))

                if 0 <= px < width and 0 <= py < height:
                    result[py, px] = [255, 0, 0]

        return result.astype(np.uint8)

    def get_kernel(self, kernel_name: str) -> np.ndarray:
        kernels = {
            'blur': np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]) / 9,

            'gaussian': np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]) / 16,

            'sharpen': np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]]),

            'edge_detect': np.array([[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]]),

            'emboss': np.array([[-2, -1, 0],
                                [-1, 1, 1],
                                [0, 1, 2]]),

            'sobel_x': np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]),

            'sobel_y': np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]]),

            'laplacian': np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]]),

            'box_blur': np.ones((5, 5)) / 25,

            'motion_blur': np.array([[1, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1]]) / 5
        }

        return kernels.get(kernel_name, kernels['gaussian'])

    def apply_convolution(self, image: np.ndarray, kernel_name: str = 'gaussian') -> np.ndarray:
        """Применяет свертку с выбранным ядром"""
        if len(image.shape) == 3:
            gray_image = self._rgb_to_grayscale(image)
        else:
            gray_image = image

        kernel = self.get_kernel(kernel_name)
        result = self._convolution(gray_image.astype(np.float32), kernel)

        # Нормализуем результат для отображения
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def apply_color_convolution(self, image: np.ndarray, kernel_name: str = 'gaussian') -> np.ndarray:
        """Применяет свертку к цветному изображению (сохраняет цвет)"""
        if len(image.shape) != 3 or image.shape[2] != 3:
            # Если изображение не цветное, используем обычную свертку
            return self.apply_convolution(image, kernel_name)

        kernel = self.get_kernel(kernel_name)
        result = np.zeros_like(image, dtype=np.float32)

        # Применяем свертку к каждому каналу отдельно
        for channel in range(3):
            result[:, :, channel] = self._convolution(
                image[:, :, channel].astype(np.float32),
                kernel
            )

        # Нормализуем результат
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
