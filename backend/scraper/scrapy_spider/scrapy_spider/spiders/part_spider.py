import scrapy
from scrapy_spider.items import PartItem


class PartSpider(scrapy.Spider):
    name = "part_spider"
    allowed_domains = ["partselect.com"]  # Replace with the target domain
    start_urls = [
        "https://www.partselect.com/Refrigerator-Parts.htm",
        "https://www.partselect.com/Dishwasher-Parts.htm",
    ]

    def parse(self, response):
        # Determine appliance type based on the start URL
        if "Refrigerator-Parts" in response.url:
            appliance_type = "refrigerator"
        elif "Dishwasher-Parts" in response.url:
            appliance_type = "dishwasher"
        else:
            appliance_type = "unknown"

        brands_list = response.css("ul.nf__links")
        brand_links = brands_list.css("li a::attr(href)").getall()

        for link in brand_links:
            yield response.follow(
                link, self.parse_brand, meta={"appliance": appliance_type}
            )

    def parse_brand(self, response):
        # Extract links to individual parts
        part_links = response.css("a.nf__part__detail__title::attr(href)").getall()
        for link in part_links:
            # Follow each link to parse individual part details
            yield response.follow(
                link, self.parse_part, meta={"appliance": response.meta["appliance"]}
            )

    def parse_part(self, response):
        item = PartItem()

        item["part_name"] = response.css("h1::text").get()

        # Alternatively, using Scrapy's re_first method
        item["part_number"] = response.css(
            '.mt-3.mb-2:contains("PartSelect Number") ::text'
        ).re_first(r"PS\d+")

        # 3. manufacturer
        item["manufacturer"] = response.css(
            '.mb-2:contains("Manufactured by") ::text'
        ).re_first(r"LG|Kenmore")

        # 4. price
        item["price"] = response.css(".price::text").re_first(r"\d+\.\d+")

        # # 5. availability: TODO
        # item["availability"] = response.css(".js-partAvailability::text").get()

        # 6. description
        item["description"] = " ".join(
            response.css('div[itemprop="description"]::text').getall()
        ).strip()

        # 7. rating
        item["rating"] = response.css(".rating__stars__upper::attr(style)").re_first(
            r"\d+"
        )

        # Extract the 'Troubleshooting' section
        # Locate the 'Troubleshooting' section using XPath
        troubleshooting_section = response.xpath(
            '//div[@id="Troubleshooting"]/following-sibling::div[@data-collapsible][1]'
        )

        # Extract all text within this section
        troubleshooting_text_list = troubleshooting_section.xpath(".//text()").getall()

        # Clean the text
        troubleshooting_text_list = [
            text.strip() for text in troubleshooting_text_list if text.strip()
        ]

        # Combine the text into a single string
        item["troubleshooting_tips"] = " ".join(troubleshooting_text_list)

        item["review_count"] = response.css(".rating__count::text").re_first(r"\d+")

        # Determine appliance type based on starting URL
        item["appliance"] = response.meta["appliance"]

        compatible_models = response.css(
            "div.pd__crossref__list.js-dataContainer.js-infiniteScroll div.row a::text"
        ).getall()
        # Clean and normalize the model numbers
        compatible_models = [
            model.strip().upper() for model in compatible_models if model.strip()
        ]
        item["compatible_models"] = compatible_models

        yield item
