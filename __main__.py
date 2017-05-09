__author__ = 'MA573RWARR10R'
import pandas as pd
import numpy as np
from AufZumHimmel import Sturmer


def _not_nan(w):
    return not pd.isnull(w)


def make_sep_file(df, necessary_attribute):
    ndf = df[[necessary_attribute]]
    ndf_fname = necessary_attribute + '.csv'
    try:
        ndf.to_csv(ndf_fname)
        return True
    except Exception as e:
        print(e)
        return False


def make_separate_attributes_files(all_attrs_file_name, necessary_attrs_as_y):
    # make separate files; each will look like ID - Attribute_name

    all_data = pd.DataFrame.from_csv(all_attrs_file_name)

    for attribute in necessary_attrs_as_y:
        if make_sep_file(all_data, attribute):
            print("Succeed with [ {} ]".format(attribute))
        else:
            print("Failure with [ {} ]".format(attribute))


def classes_names_to_ids(fname, prefix="classes_"):
    df = pd.DataFrame.from_csv(fname)
    feature_name = fname.split(".")[0]

    try:
        all_words = [w for w in filter(_not_nan, df[feature_name].tolist())]
        all_w_series = pd.Series(all_words)
        all_w_series = all_w_series.unique()

        classes_ids_names = [(cname, cid) for (cname, cid) in zip(all_w_series, range(1, len(all_w_series)+1))]

        classes_ids_names = pd.DataFrame(classes_ids_names, columns=['class_name', 'class_id'])
        classes_ids_names.to_csv(prefix + fname)

        return True

    except Exception as e:
        print(e)
        return False


def make_classes_from_attributes(train_fnames):
    for train_file in train_fnames:
        if classes_names_to_ids(train_file):
            print("Succeeded with [ {} ]".format(train_file))
        else:
            print("Failure with [ {} ]".format(train_file))


def mark_files_acc_class_ids(classes_files):
    def set_id(_row, _class_name_dict):
        try:
            return _class_name_dict[_row]
        except KeyError:
            print(_row)

    for class_f in classes_files:
        try:
            classes_df = pd.DataFrame.from_csv(class_f)
            class_name = class_f.split("_")[1].split(".")[0]

            classes_names_ids_dict = dict(zip(classes_df.class_name, classes_df.class_id))

            all_data_df = pd.DataFrame.from_csv('product_attributes.csv')

            data_feature_slice = all_data_df[[class_name]]
            data_feature_slice.dropna(inplace=True)
            data_feature_slice = data_feature_slice[class_name].apply(set_id, args=(classes_names_ids_dict,))
            data_feature_slice_df = pd.DataFrame(data=data_feature_slice)
            data_feature_slice_df.rename(columns={class_name: 'class_id'}, inplace=True)

            data_feature_slice_df.to_csv('product_id_class_id_' + class_name + '.csv')

            print("Success with [ {} ]".format(class_name))

        except Exception as e:
            print(e)
            print("Failure with [ {} ]".format(class_f))


def products_to_classes_ids(classes_file_names, remark_ids=False):
    if remark_ids:
        mark_files_acc_class_ids(classes_file_names)


def get_file_names(feature):
    return 'product_id_class_id_' + feature + '.csv', 'classes_' + feature + '.csv', feature

if __name__ == "__main__":
    fname = 'product_attributes.csv'
    nc = ['Type', 'Finish1', 'Treatment', 'Hanging Method', 'Shade Color', 'Shade Shape', 'Tier']

    # make_separate_attributes_files(all_attrs_file_name=fname, necessary_attrs_as_y=nc)
    train_fnames = [fna + '.csv' for fna in nc]
    classes_fnames = ['classes_' + fna + '.csv' for fna in nc]

    # make_classes_from_attributes(train_fnames)
    # products_to_classes_ids(classes_fnames, True)

    product_class_files = ['product_id_class_id_' + fna + '.csv' for fna in nc]
    train_files = ["features/features_" + fna + '.csv' for fna in nc]

    f = nc[0]
    im_filepath = 'ml/'
    s = Sturmer(*get_file_names(f), images_file_path=im_filepath)
    # s._generate_more()
    # s._images_to_vectors(parts=1)
    s.build_model(train_files[0])
