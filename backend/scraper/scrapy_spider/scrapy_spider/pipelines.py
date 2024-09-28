# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter


# class ScrapySpiderPipeline:
#     def process_item(self, item, spider):
#         return item
import json
import os


class JsonWriterPipeline:
    def open_spider(self, spider):
        data_dir = os.path.join(os.path.dirname(__file__), "../../../data")
        os.makedirs(data_dir, exist_ok=True)
        self.file = open(os.path.join(data_dir, "parts_data.json"), "w")
        self.file.write("[")
        self.first_item = True

    def close_spider(self, spider):
        self.file.write("]")
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item), ensure_ascii=False)
        if self.first_item:
            self.first_item = False
        else:
            self.file.write(",\n")
        self.file.write(line)
        return item
