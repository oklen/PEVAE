import pickle
import argparse
import os

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, required=True,help='Sepcify Where data store !')
    args.add_argument('--output_dirs', type=str, required=True, help='Where to output result.')
    args.add_argument('--k_core',type=int, default=5)
    args = args.parse_args()

    with open(args.data_path,'rb') as f:
        reviews = pickle.load(f)
        def check_sat(dic):
            to_remove = []
            for key in dic.keys():
                if dic[key] < args.k_core:
                    to_remove.append(key)
            return to_remove
        while True:
            user_dict = {}
            item_dict = {}
            for review in reivews:
                user_id = review['user']
                item_id = review['item']
                if user_dict.get(user_id) is None:
                    user_dict[user_dict] = 1
                else:
                    user_dict[user_dict] += 1
                
                if item_dict.get(item_id) is None:
                    item_dict[item_id] = 1
                else:
                    item_dict[item_id] += 1
            user_to_remove = check_sat(user_dict)
            item_to_remove = check_sat(item_dict)
            if len(user_to_remove) == 0 and len(item_to_remove) == 0:
                break
            new_reviews = []
            for review in reviews:
                if review['user'] in user_to_remove or review['item'] item_to_remove:
                    continue
                new_reviews = review.append(review)
            reviews = new_review
    input_filename = os.path.split(args.data_path)[-1]
    save_file_name = os.path.join(args.output_dirs, os.path.splitext(input_filename)[0] + '_' + str(args.k_core) + '.pkl')
    with open(save_file_name,'wb') as f:
        pickle.dump(reviews, f)
    print("Generate done!")


        