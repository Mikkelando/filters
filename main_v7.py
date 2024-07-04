import csv
import os

import numpy as np
from filter_utils import smooth_lnd_for_video, get_video
import mediapipe as mp
import cv2
from tqdm import tqdm

def get_face_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    else:
        return None


def draw_landmarks_on_frames(frames, landmarks_list):
    frames_with_landmarks = []
    for frame, landmarks in zip(frames, landmarks_list):
        frame_copy = frame.copy()
        for (x, y) in landmarks:
            cv2.circle(frame_copy, (x, y), 2, (0, 255, 0), -1)
        frames_with_landmarks.append(frame_copy)
    return frames_with_landmarks


def save_frames_to_video(frames, output_path, fps=25):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()


def write_to_csv( data, file_path='data/lnd.csv', headers=None, qnt=468):
   
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            headers = [f'x_{i}' for i in range(qnt)] + [f'y_{i}' for i in range(qnt)]
 
            writer.writerow(headers)

        writer.writerow(data)


if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    
    gen_imgs = get_video('data/karin_out_00078_t0.2_final.mp4')


    LND = []

    
    for gen_img in tqdm(gen_imgs):

        gen_landmarks = get_face_landmarks(gen_img, face_mesh)
        

        gen_landmarks = [(int(landmark.x * gen_img.shape[1]), int(landmark.y * gen_img.shape[0])) for landmark in gen_landmarks]
        LND.append(gen_landmarks)

    print(np.array(LND).shape)    

    # for lnd in LND:
    #     xs = np.array(lnd)[:, 0]
    #     ys = np.array(lnd)[:, 1]
    #     data = np.concatenate((xs, ys))
    #     write_to_csv(data, qnt=len(xs))
        
    print('PROCESS')

   

    frames_names = ['subdata/f_0.jpg', 'subdata/f_1.jpg', 'subdata/f_2.jpg', 'subdata/f_3.jpg', 'subdata/f_4.jpg', 'subdata/f_5.jpg', 'subdata/f_6.jpg', 'subdata/f_7.jpg', 'subdata/f_8.jpg', 'subdata/f_9.jpg', 'subdata/f_10.jpg', 'subdata/f_11.jpg', 'subdata/f_12.jpg', 'subdata/f_13.jpg', 'subdata/f_14.jpg', 'subdata/f_15.jpg', 'subdata/f_16.jpg', 'subdata/f_17.jpg', 'subdata/f_18.jpg', 'subdata/f_19.jpg', 'subdata/f_20.jpg', 'subdata/f_21.jpg', 'subdata/f_22.jpg', 'subdata/f_23.jpg', 'subdata/f_24.jpg', 'subdata/f_25.jpg', 'subdata/f_26.jpg', 'subdata/f_27.jpg', 'subdata/f_28.jpg', 'subdata/f_29.jpg', 'subdata/f_30.jpg', 'subdata/f_31.jpg', 'subdata/f_32.jpg', 'subdata/f_33.jpg', 'subdata/f_34.jpg', 'subdata/f_35.jpg', 'subdata/f_36.jpg', 'subdata/f_37.jpg', 'subdata/f_38.jpg', 'subdata/f_39.jpg', 'subdata/f_40.jpg', 'subdata/f_41.jpg', 'subdata/f_42.jpg', 'subdata/f_43.jpg', 'subdata/f_44.jpg', 'subdata/f_45.jpg', 'subdata/f_46.jpg', 'subdata/f_47.jpg', 'subdata/f_48.jpg', 'subdata/f_49.jpg', 'subdata/f_50.jpg', 'subdata/f_51.jpg', 'subdata/f_52.jpg', 'subdata/f_53.jpg', 'subdata/f_54.jpg', 'subdata/f_55.jpg', 'subdata/f_56.jpg', 'subdata/f_57.jpg', 'subdata/f_58.jpg', 'subdata/f_59.jpg', 'subdata/f_60.jpg', 'subdata/f_61.jpg', 'subdata/f_62.jpg', 'subdata/f_63.jpg', 'subdata/f_64.jpg', 'subdata/f_65.jpg', 'subdata/f_66.jpg', 'subdata/f_67.jpg', 'subdata/f_68.jpg', 'subdata/f_69.jpg', 'subdata/f_70.jpg', 'subdata/f_71.jpg', 'subdata/f_72.jpg', 'subdata/f_73.jpg', 'subdata/f_74.jpg', 'subdata/f_75.jpg', 'subdata/f_76.jpg', 'subdata/f_77.jpg', 'subdata/f_78.jpg', 'subdata/f_79.jpg', 'subdata/f_80.jpg', 'subdata/f_81.jpg', 'subdata/f_82.jpg', 'subdata/f_83.jpg', 'subdata/f_84.jpg', 'subdata/f_85.jpg', 'subdata/f_86.jpg', 'subdata/f_87.jpg', 'subdata/f_88.jpg', 'subdata/f_89.jpg', 'subdata/f_90.jpg', 'subdata/f_91.jpg', 'subdata/f_92.jpg', 'subdata/f_93.jpg', 'subdata/f_94.jpg', 'subdata/f_95.jpg', 'subdata/f_96.jpg', 'subdata/f_97.jpg', 'subdata/f_98.jpg', 'subdata/f_99.jpg', 'subdata/f_100.jpg', 'subdata/f_101.jpg', 'subdata/f_102.jpg', 'subdata/f_103.jpg', 'subdata/f_104.jpg', 'subdata/f_105.jpg', 'subdata/f_106.jpg', 'subdata/f_107.jpg', 'subdata/f_108.jpg', 'subdata/f_109.jpg', 'subdata/f_110.jpg', 'subdata/f_111.jpg', 'subdata/f_112.jpg', 'subdata/f_113.jpg', 'subdata/f_114.jpg', 'subdata/f_115.jpg', 'subdata/f_116.jpg', 'subdata/f_117.jpg', 'subdata/f_118.jpg', 'subdata/f_119.jpg', 'subdata/f_120.jpg', 'subdata/f_121.jpg', 'subdata/f_122.jpg', 'subdata/f_123.jpg', 'subdata/f_124.jpg', 'subdata/f_125.jpg', 'subdata/f_126.jpg', 'subdata/f_127.jpg', 'subdata/f_128.jpg', 'subdata/f_129.jpg', 'subdata/f_130.jpg', 'subdata/f_131.jpg', 'subdata/f_132.jpg', 'subdata/f_133.jpg', 'subdata/f_134.jpg', 'subdata/f_135.jpg', 'subdata/f_136.jpg', 'subdata/f_137.jpg', 'subdata/f_138.jpg', 'subdata/f_139.jpg', 'subdata/f_140.jpg', 'subdata/f_141.jpg', 'subdata/f_142.jpg', 'subdata/f_143.jpg', 'subdata/f_144.jpg', 'subdata/f_145.jpg', 'subdata/f_146.jpg', 'subdata/f_147.jpg', 'subdata/f_148.jpg', 'subdata/f_149.jpg', 'subdata/f_150.jpg', 'subdata/f_151.jpg', 'subdata/f_152.jpg', 'subdata/f_153.jpg', 'subdata/f_154.jpg', 'subdata/f_155.jpg', 'subdata/f_156.jpg', 'subdata/f_157.jpg', 'subdata/f_158.jpg', 'subdata/f_159.jpg', 'subdata/f_160.jpg', 'subdata/f_161.jpg', 'subdata/f_162.jpg', 'subdata/f_163.jpg', 
        'subdata/f_164.jpg', 'subdata/f_165.jpg', 'subdata/f_166.jpg', 'subdata/f_167.jpg', 'subdata/f_168.jpg', 'subdata/f_169.jpg', 'subdata/f_170.jpg', 'subdata/f_171.jpg', 'subdata/f_172.jpg', 'subdata/f_173.jpg', 'subdata/f_174.jpg', 'subdata/f_175.jpg', 'subdata/f_176.jpg', 'subdata/f_177.jpg', 'subdata/f_178.jpg', 'subdata/f_179.jpg', 'subdata/f_180.jpg', 'subdata/f_181.jpg', 'subdata/f_182.jpg', 'subdata/f_183.jpg', 'subdata/f_184.jpg', 'subdata/f_185.jpg', 'subdata/f_186.jpg', 'subdata/f_187.jpg', 'subdata/f_188.jpg', 'subdata/f_189.jpg', 'subdata/f_190.jpg', 'subdata/f_191.jpg', 'subdata/f_192.jpg', 'subdata/f_193.jpg', 'subdata/f_194.jpg', 'subdata/f_195.jpg', 'subdata/f_196.jpg', 'subdata/f_197.jpg', 'subdata/f_198.jpg', 'subdata/f_199.jpg', 'subdata/f_200.jpg', 'subdata/f_201.jpg', 'subdata/f_202.jpg', 'subdata/f_203.jpg', 'subdata/f_204.jpg', 'subdata/f_205.jpg', 'subdata/f_206.jpg', 'subdata/f_207.jpg', 'subdata/f_208.jpg', 'subdata/f_209.jpg', 'subdata/f_210.jpg', 'subdata/f_211.jpg', 'subdata/f_212.jpg', 'subdata/f_213.jpg', 'subdata/f_214.jpg', 'subdata/f_215.jpg', 'subdata/f_216.jpg', 'subdata/f_217.jpg', 'subdata/f_218.jpg', 'subdata/f_219.jpg', 'subdata/f_220.jpg', 'subdata/f_221.jpg', 'subdata/f_222.jpg', 'subdata/f_223.jpg', 'subdata/f_224.jpg', 'subdata/f_225.jpg', 'subdata/f_226.jpg', 'subdata/f_227.jpg', 'subdata/f_228.jpg', 'subdata/f_229.jpg', 'subdata/f_230.jpg', 'subdata/f_231.jpg', 'subdata/f_232.jpg', 'subdata/f_233.jpg', 'subdata/f_234.jpg', 'subdata/f_235.jpg', 'subdata/f_236.jpg', 'subdata/f_237.jpg', 'subdata/f_238.jpg', 'subdata/f_239.jpg', 'subdata/f_240.jpg', 'subdata/f_241.jpg', 'subdata/f_242.jpg', 'subdata/f_243.jpg', 'subdata/f_244.jpg', 'subdata/f_245.jpg', 'subdata/f_246.jpg', 'subdata/f_247.jpg', 'subdata/f_248.jpg', 'subdata/f_249.jpg', 'subdata/f_250.jpg', 'subdata/f_251.jpg', 'subdata/f_252.jpg', 'subdata/f_253.jpg', 'subdata/f_254.jpg', 'subdata/f_255.jpg', 'subdata/f_256.jpg', 'subdata/f_257.jpg', 'subdata/f_258.jpg', 'subdata/f_259.jpg', 'subdata/f_260.jpg', 'subdata/f_261.jpg', 'subdata/f_262.jpg', 'subdata/f_263.jpg', 'subdata/f_264.jpg', 'subdata/f_265.jpg', 'subdata/f_266.jpg', 'subdata/f_267.jpg', 'subdata/f_268.jpg', 'subdata/f_269.jpg', 'subdata/f_270.jpg', 'subdata/f_271.jpg', 'subdata/f_272.jpg', 'subdata/f_273.jpg', 'subdata/f_274.jpg', 'subdata/f_275.jpg', 'subdata/f_276.jpg', 'subdata/f_277.jpg', 'subdata/f_278.jpg', 'subdata/f_279.jpg', 'subdata/f_280.jpg', 'subdata/f_281.jpg', 'subdata/f_282.jpg', 'subdata/f_283.jpg', 'subdata/f_284.jpg', 'subdata/f_285.jpg', 'subdata/f_286.jpg', 'subdata/f_287.jpg', 'subdata/f_288.jpg', 'subdata/f_289.jpg', 'subdata/f_290.jpg', 'subdata/f_291.jpg', 'subdata/f_292.jpg', 'subdata/f_293.jpg', 'subdata/f_294.jpg', 'subdata/f_295.jpg', 'subdata/f_296.jpg', 'subdata/f_297.jpg', 'subdata/f_298.jpg', 'subdata/f_299.jpg', 'subdata/f_300.jpg', 'subdata/f_301.jpg', 'subdata/f_302.jpg', 'subdata/f_303.jpg', 'subdata/f_304.jpg', 'subdata/f_305.jpg', 'subdata/f_306.jpg', 'subdata/f_307.jpg', 'subdata/f_308.jpg', 'subdata/f_309.jpg', 'subdata/f_310.jpg', 'subdata/f_311.jpg', 'subdata/f_312.jpg', 'subdata/f_313.jpg', 'subdata/f_314.jpg', 'subdata/f_315.jpg', 'subdata/f_316.jpg', 'subdata/f_317.jpg', 
        'subdata/f_318.jpg', 'subdata/f_319.jpg', 'subdata/f_320.jpg', 'subdata/f_321.jpg', 'subdata/f_322.jpg', 'subdata/f_323.jpg', 'subdata/f_324.jpg', 'subdata/f_325.jpg', 'subdata/f_326.jpg', 'subdata/f_327.jpg', 'subdata/f_328.jpg', 'subdata/f_329.jpg', 'subdata/f_330.jpg', 'subdata/f_331.jpg', 'subdata/f_332.jpg', 'subdata/f_333.jpg', 'subdata/f_334.jpg', 'subdata/f_335.jpg', 'subdata/f_336.jpg', 'subdata/f_337.jpg', 'subdata/f_338.jpg', 'subdata/f_339.jpg', 'subdata/f_340.jpg', 'subdata/f_341.jpg', 'subdata/f_342.jpg', 'subdata/f_343.jpg', 'subdata/f_344.jpg', 'subdata/f_345.jpg', 'subdata/f_346.jpg', 'subdata/f_347.jpg', 'subdata/f_348.jpg', 'subdata/f_349.jpg', 'subdata/f_350.jpg', 'subdata/f_351.jpg', 'subdata/f_352.jpg', 'subdata/f_353.jpg', 'subdata/f_354.jpg', 'subdata/f_355.jpg', 'subdata/f_356.jpg', 'subdata/f_357.jpg', 'subdata/f_358.jpg', 'subdata/f_359.jpg', 'subdata/f_360.jpg', 'subdata/f_361.jpg', 'subdata/f_362.jpg', 'subdata/f_363.jpg', 'subdata/f_364.jpg', 'subdata/f_365.jpg', 'subdata/f_366.jpg', 'subdata/f_367.jpg', 'subdata/f_368.jpg', 'subdata/f_369.jpg', 'subdata/f_370.jpg', 'subdata/f_371.jpg', 'subdata/f_372.jpg', 'subdata/f_373.jpg', 'subdata/f_374.jpg', 'subdata/f_375.jpg', 'subdata/f_376.jpg', 'subdata/f_377.jpg', 'subdata/f_378.jpg', 'subdata/f_379.jpg', 'subdata/f_380.jpg', 'subdata/f_381.jpg', 'subdata/f_382.jpg', 'subdata/f_383.jpg', 'subdata/f_384.jpg', 'subdata/f_385.jpg', 'subdata/f_386.jpg', 'subdata/f_387.jpg', 'subdata/f_388.jpg', 'subdata/f_389.jpg', 'subdata/f_390.jpg', 'subdata/f_391.jpg', 'subdata/f_392.jpg', 'subdata/f_393.jpg', 'subdata/f_394.jpg', 'subdata/f_395.jpg', 'subdata/f_396.jpg', 'subdata/f_397.jpg', 'subdata/f_398.jpg', 'subdata/f_399.jpg', 'subdata/f_400.jpg', 'subdata/f_401.jpg', 'subdata/f_402.jpg', 'subdata/f_403.jpg', 'subdata/f_404.jpg', 'subdata/f_405.jpg', 'subdata/f_406.jpg', 'subdata/f_407.jpg', 'subdata/f_408.jpg', 'subdata/f_409.jpg', 'subdata/f_410.jpg', 'subdata/f_411.jpg', 'subdata/f_412.jpg', 'subdata/f_413.jpg', 'subdata/f_414.jpg', 'subdata/f_415.jpg', 'subdata/f_416.jpg', 'subdata/f_417.jpg', 'subdata/f_418.jpg', 'subdata/f_419.jpg', 'subdata/f_420.jpg', 'subdata/f_421.jpg', 'subdata/f_422.jpg', 'subdata/f_423.jpg', 'subdata/f_424.jpg', 'subdata/f_425.jpg', 'subdata/f_426.jpg', 'subdata/f_427.jpg', 'subdata/f_428.jpg', 'subdata/f_429.jpg', 'subdata/f_430.jpg', 'subdata/f_431.jpg', 'subdata/f_432.jpg', 'subdata/f_433.jpg', 'subdata/f_434.jpg', 'subdata/f_435.jpg', 'subdata/f_436.jpg', 'subdata/f_437.jpg', 'subdata/f_438.jpg', 'subdata/f_439.jpg', 'subdata/f_440.jpg', 'subdata/f_441.jpg', 'subdata/f_442.jpg', 'subdata/f_443.jpg', 'subdata/f_444.jpg', 'subdata/f_445.jpg', 'subdata/f_446.jpg', 'subdata/f_447.jpg', 'subdata/f_448.jpg', 'subdata/f_449.jpg', 'subdata/f_450.jpg', 'subdata/f_451.jpg', 'subdata/f_452.jpg', 'subdata/f_453.jpg', 'subdata/f_454.jpg', 'subdata/f_455.jpg', 'subdata/f_456.jpg', 'subdata/f_457.jpg', 'subdata/f_458.jpg', 'subdata/f_459.jpg']
    
    lnd_names = 'data/lnd.csv'

    NEW_LND = smooth_lnd_for_video(frames_names, lnd_names, power =7, fps = 25, qnt_l=468)


    frames_with_landmarks_1 = draw_landmarks_on_frames(gen_imgs, LND)
    frames_with_landmarks_2 = draw_landmarks_on_frames(gen_imgs, NEW_LND)
    save_frames_to_video(frames_with_landmarks_1, 'data/orig.mp4')
    save_frames_to_video(frames_with_landmarks_2, 'data/smooth.mp4')
