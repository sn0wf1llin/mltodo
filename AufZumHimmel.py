__author__ = 'MA573RWARR10R'
import os
import csv

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import pandas as pd
import numpy as np
from split_dataset import _splitter
import xgboost


class Sturmer:
    def __init__(self, product_class_file, classes_names_ids_file,
                 feature_name, images_file_path, generated_images_file_path="generated_images/"):

        self.generated_images_file_path = generated_images_file_path + feature_name + "/"
        self.product_class_file = product_class_file
        self.classes_names_ids_file = classes_names_ids_file
        self.feature_to_predict = feature_name
        self.images_file_path = images_file_path
        self.data_gen = None
        self.multi_class_classifier_model = None

    def _make_image_data_generator(self, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rescale=1. / 255,
                                   shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest'):
        self.data_gen = ImageDataGenerator(
            rotation_range=rotation_range, width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            rescale=rescale,
            shear_range=shift_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode)

    def _generate_more(self, generated_file_path=None, image_new_shape=(100, 100)):
        if generated_file_path is not None:
            self.generated_images_file_path = generated_file_path

        if not os.path.exists(self.generated_images_file_path):
            os.makedirs(self.generated_images_file_path)

        if self.data_gen is None:
            self._make_image_data_generator()

        df_product_classes = pd.DataFrame.from_csv(self.product_class_file)

        if self.data_gen is None:
            raise Exception("Need to create datagen first. [ make_image_data_generator ]\nExiting...\n")

        for row in df_product_classes.iterrows():
            product_id = row[0]
            class_id = row[1]['class_id']

            if self.images_file_path[-1] != "/":
                self.images_file_path += "/"

            try:
                img = load_img(self.images_file_path + product_id + ".jpg", target_size=image_new_shape)

                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                i = 0
                for batch in self.data_gen.flow(x, batch_size=1, save_to_dir=self.generated_images_file_path,
                                                save_prefix=str(class_id) + product_id, save_format='jpeg'):
                    i += 1
                    if i > 2:
                        break

            except OSError:
                pass

    def _images_to_vectors(self, parts=None, images_features_save_path="features/", image_new_shape=(100, 100)):
        if not os.path.exists(images_features_save_path):
            os.makedirs(images_features_save_path)

        images_vector_file_path = images_features_save_path + "features_" + self.feature_to_predict + ".csv"

        def _process_image(image_path, image_class_id, tsize):
            try:
                img = load_img(image_path, target_size=tsize)

                img_arr = img_to_array(img)
                img_arr = img_arr.reshape((img_arr.shape[0] * img_arr.shape[1] * img_arr.shape[2]))
                img_arr = np.append(img_arr, image_class_id)

                return img_arr
            except (FileNotFoundError, OSError):
                return None

        def _save_part_to_csv(part_to_save, is_columns=False):
            try:
                with open(images_vector_file_path, "a") as f:
                    writer = csv.writer(f, dialect='excel')
                    if is_columns:
                        writer.writerow(part_to_save)
                    else:
                        writer.writerows(part_to_save)

                    print('Successfully saved as {}'.format(images_vector_file_path))

            except Exception as e:
                print("SOMETHING WRONG WITH SAVING FEATURES")
                print(e)

        df_product_classes = pd.DataFrame.from_csv(self.product_class_file)
        images_files_count = len(os.listdir(self.images_file_path))

        f_count = 3 * image_new_shape[0] * image_new_shape[1]
        _cols = ['f' + str(i) for i in range(f_count)] + ['target']
        _save_part_to_csv(_cols, is_columns=True)

        _ = []
        images_counter = 0
        parts_counter = 0

        for row in df_product_classes.iterrows():
            product_id = row[0]
            class_id = row[1]['class_id']

            im_res = _process_image(image_path=self.images_file_path + product_id + ".jpg", image_class_id=class_id,
                                    tsize=image_new_shape)

            if im_res is not None:
                _.append(im_res)
                images_counter += 1
                print("{} Class [ {} ] appended".format(images_counter, class_id))
            else:
                print("[ {} ] has troubles".format(product_id))

            if images_counter == int(images_files_count * 1 / 10) and parts_counter != parts:
                parts_counter += 1
                images_counter = 0
                _save_part_to_csv(_)
                _ = []

        if parts_counter is None:
            _save_part_to_csv(_)

        del _
        # images_features_df = pd.DataFrame.from_csv(images_vector_file_path, header=1)
        #
        # try:
        #     images_features_df.to_csv(images_features_save_path + "features_" + self.feature_to_predict + ".csv")
        # except Exception as e:
        #     print(e)
        #     print("Can't save file [ {} ]".format(
        #         images_features_save_path + "features_" + self.feature_to_predict + ".csv"))

    @staticmethod
    def _get_multi_class_classifier_parameters(classes_count, random_seed=88):
        # silent                - work mode; 1 - silent activated
        # nthread               - count of threads; default to maximum available count

        general_parameters = {
            'silent': 0,
        }

        # learning_rate         - analogous to eta
        # min_child_weight      - default = 1; used to prevent over-fitting; to high values -> under-fitting
        # max_depth             - max depth of a tree; typical 3-10
        # max_leaf_nodes        - defined with help of max_depth as 2^n
        # gamma                 - min value of loss function to make a split
        # max_delta_step        - updating each tree weights
        # subsample             - fraction of observations for each tree; typical 0.5-1
        # colsample_bytree      - fraction of columns for each tree; typical 0.5-1
        # colsample_bylevel     --> all job made by 2 previous
        # reg_lambda            - l2 regularization parameter
        # reg_alpha             - l1 regularization parameter; used with high dimensional data
        # scale_pos_weight      - used in case of high class imbalance
        # n_estimators          - number of boosted trees to fit
        # base_score            - initial prediction score of all instances, global bias
        # missing               - value in the data which needs to be present as a missing value. None defaults to np.nan

        booster_parameters = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'gamma': 0.01,
            'max_delta_step': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 100,
            'reg_alpha': 8,

        }

        # objective             - loss function to be minimized;
        #   -- binary:logistic  -> logistic regression;returns predicted probability (not class)
        #   -- multi:softmax    -> multiclass classification;returns predicted class (not probability); requires num_class
        #   -- multi:softprob   -> same as softmax but returns probabilities belonging to each class
        # eval_metric           - default according to objective;metric used for validation data
        #   -- rmse             –> root mean square error
        #   -- mae              –> mean absolute error
        #   -- logloss          –> negative log-likelihood
        #   -- error            –> binary classification error rate (0.5 threshold)
        #   -- merror           –> multiclass classification error rate
        #   -- mlogloss         –> multiclass logloss
        #   -- auc              -> area under the curve
        # seed                  - random number seed
        # num_class             -> number of classes

        lt_parameters = {
            'objective': 'multi:softmax',
            'num_class': classes_count,
            'seed': random_seed,
        }

        all_params = {}
        all_params.update(general_parameters)
        all_params.update(booster_parameters)
        all_params.update(lt_parameters)

        return all_params

    def build_model(self, train_file):
        train_df, test_df = _splitter(train_file, dtype='unicode')
        sz = train_df.shape[1]

        train_X = train_df[:, 0:sz - 1]
        train_Y = train_df[:, sz]

        test_X = test_df[:, 0:sz - 1]
        test_Y = test_df[:, sz]

        xg_train = xgboost.DMatrix(train_X, label=train_Y)
        xg_test = xgboost.DMatrix(test_X, label=test_Y)
        watchlist = [(xg_train, 'train'), (xg_test, 'test')]

        num_round = 5
        num_classes = max(pd.DataFrame.from_csv(self.classes_names_ids_file)['class_id'].tolist())

        multi_class_classifier_parameters = self._get_multi_class_classifier_parameters(classes_count=num_classes)
        mcl_model = xgboost.train(multi_class_classifier_parameters, xg_train, num_round, watchlist)
        self.multi_class_classifier_model = mcl_model

        pred = mcl_model.predict(xg_test)

        print('predicting, classification error=%f' % (
            sum(int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))

    def _load_model(self):
        pass

    def predict(self, image_as_vector):
        pass
