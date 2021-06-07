import techscrape as ts
from pprint import pprint

if __name__ == '__main__':
    # crawl the web with a given list of connections and a query
    # below, for example, we are looking through sifted.eu articles on
    # the topic of machine learning in healthcare
    scraper = ts.TechScraper(crawlers=[ts.Sifted])
    
    # this returns a list of company names (+ a lot of noise which we currently
    # still need to inspect manually)
    companies = scraper.get('machine learning in healthcare')
    pprint(companies)
    
    # the easier approach is the focused scraping of a specific blog post
    # which we can then use to scrape CrunchBase
    cat_comp = ts.MLBlog.search()

    cmp = []  # make a list of company names
    for cmp_ls in cat_comp.values():
        cmp += cmp_ls

    # scrape CrunchBase for companies
    data = scraper.extend_data({}, cmp)

    # we now have a dictionary of companies and their characteristics
    # we can save this as a json file
    ts.to_json(data, 'company_data.json')

    # alternatively, we can transform the data into a data-frame to save
    # the data in a `excel-friendly` format
    data_frame = ts.to_dataframe(data)
    print(  # take a look at the top 5 table rows
        data_frame.head()
    )
    data_frame.to_excel('company_data.xlsx')
