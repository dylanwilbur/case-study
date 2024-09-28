# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class PartItem(scrapy.Item):
    part_name = scrapy.Field()
    part_number = scrapy.Field()
    manufacturer = scrapy.Field()
    description = scrapy.Field()
    compatible_models = scrapy.Field()
    installation_instructions = scrapy.Field()
    troubleshooting_tips = scrapy.Field()
    price = scrapy.Field()
    availability = scrapy.Field()
    rating = scrapy.Field()
    review_count = scrapy.Field()
    reviews = scrapy.Field()
    appliance = scrapy.Field()
