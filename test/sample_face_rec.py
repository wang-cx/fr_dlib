import os
import sys

import cv2
from img_utils.files import images_in_dir, filename
from img_utils.images import put_text

from fr import detect_faces, compute_face_descriptor, closest_one, load_from_pickle


def main(samples_dir, test_dir, output_dir):
    # face_descriptors, class_names = load_samples_descriptors(samples_dir)
    # save2pickle(face_descriptors, class_names, "wg_colleagues.pkl")
    face_descriptors, class_names = load_from_pickle('wg_colleagues.pkl')
    print(face_descriptors[0])
    print("len face_descriptors: {}".format(len(face_descriptors)))
    print("len class_names: {}".format(len(class_names)))
    print("class_names: {}".format(class_names))
    image_files = images_in_dir(test_dir)
    for im_f in image_files:
        output_path = os.path.join(output_dir, filename(im_f))
        im = cv2.imread(im_f)
        faces = detect_faces(im)
        for face in faces:
            descriptor = compute_face_descriptor(im, face)
            idx, distance = closest_one(face_descriptors, descriptor)
            txt = '{}:{}'.format(class_names[idx], distance)
            put_text(im, txt, font_face=cv2.FONT_HERSHEY_SIMPLEX)
            print('{}: {}, of distance :{} '.format(im_f, class_names[idx], distance))
        cv2.imwrite(output_path, im)


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        print('---------------------------------------------------------------------------')
        print('|-- Usage: python sample_face_rec.py ${samples_dir} ${test_data_dir} ${output_dir}')
        print('---------------------------------------------------------------------------')
        exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
