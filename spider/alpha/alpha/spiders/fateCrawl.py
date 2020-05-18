# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import re
from alpha.items import AlphaItem

class FatecrawlSpider(CrawlSpider):
    name = 'fateCrawl'
    allowed_domains = ['wall.alphacoders.com']
    start_urls = ['https://wall.alphacoders.com/search.php?search=fate&lang=Chinese']

    rules = (
        #翻页
        Rule(LinkExtractor(restrict_xpaths=["//ul[@class='pagination']/li"]), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        item = {}
        thumb_img_list = response.xpath("//div[@class='boxgrid']//img/@data-src").extract() #缩略图
        # self.logger.info(len(thumb_img_list))
        #https://images2.alphacoders.com/596/thumb-350-596296.jpeg
        item = AlphaItem()
        item['image_urls'] = [re.sub("thumb-\d+-","",i) for i in thumb_img_list]
        yield item
