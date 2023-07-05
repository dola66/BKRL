import pdb

from data.Dataset import DataSet

class Beauty(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/Amazon/Beauty/'
        self.user_record_file = 'Beauty_item_sequences.pkl'
        self.user_mapping_file = 'Beauty_user_mapping.pkl'
        self.item_mapping_file = 'Beauty_item_mapping.pkl'
        self.kg_file = 'embedding.txt'

        self.num_users = 22363
        self.num_items = 12101
        self.vocab_size = 0



        # self.dir_path = './data/dataset/Amazon/last-fm/'
        # self.user_record_file = 'jiaohu.pkl'
        # self.user_mapping_file = 'use.pkl'
        # self.item_mapping_file = 'item.pkl'
        # self.kg_file = 'lamTransE_entity_emb1.txt'
        #
        # self.num_users = 23566
        # self.num_items = 48123
        # self.vocab_size = 0

        # self.dir_path = './data/dataset/Amazon/yelp2018/'
        # self.user_record_file = 'jiaohu.pkl'
        # self.user_mapping_file = 'use.pkl'
        # self.item_mapping_file = 'item.pkl'
        # self.kg_file = 'yelpTransE_entity_emb1.txt'
        #
        # self.num_users = 45919
        # self.num_items = 45538
        # self.vocab_size = 0


        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, index_shift=1):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        kg_mapping = self.load_kg(self.dir_path+self.kg_file,self.num_items)
        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        user_records = self.data_index_shift(user_records, increase_by=index_shift)
        train_set, test_set = self.split_data_sequentially(user_records, test_radio=0.2)




        return train_set, test_set, self.num_users, self.num_items + index_shift, kg_mapping