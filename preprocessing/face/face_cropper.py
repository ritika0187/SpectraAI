class FaceCropper:

    def crop(self, image, faces):

        crops = []

        for (x, y, w, h) in faces:

            padding = int(w * 0.20)

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)

            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            face = image[y1:y2, x1:x2]

            if face.size != 0:
                crops.append(face)

        return crops