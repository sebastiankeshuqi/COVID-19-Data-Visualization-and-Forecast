#!/usr/bin/env python
# coding: utf-8

# # 1. python preparation
# ---
# * :https://news.qq.com/zt2020/page/feiyan.htm#/
# * 
# 
#     getOnsInfo?name=disease_h5&callback=jQuery341028144104595763175_1582955785410&_=1582955785411
#     
#     getOnsInfo?name=disease_other&callback=jQuery34109813744456178464_1582957974845&_=1582957974846
# 
# 
# * 
#     
#     https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5&callback=jQuery341028144104595763175_1582955785410&_=1582955785411
#     
#     https://view.inews.qq.com/g2/getOnsInfo?name=disease_other&callback=jQuery34109813744456178464_1582957974845&_=1582957974846
#     
# 

# In[2]:


import time
import json
import pandas as pd
import numpy as np


# In[3]:


def get_last_data():
    import requests
    url = r"""https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5&_={0}""".format(int(time.time()*1000))
    result = requests.get(url)
    data = json.loads(result.json()['data'])
    
    return data

def get_hist_data():
    import requests
    url = r"""https://view.inews.qq.com/g2/getOnsInfo?name=disease_other&_={0}""".format(int(time.time()*1000))
    result = requests.get(url)
    data = json.loads(result.json()['data'])
    
    return data

def get_foreign_data():
    import requests
    url = r"""https://view.inews.qq.com/g2/getOnsInfo?name=disease_foreign&_={0}""".format(int(time.time()*1000))
    result = requests.get(url)
    data = json.loads(result.json()['data'])
    
    return data


# In[11]:


last_data = get_last_data()


# In[5]:


def get_city_last_info(data, province):
    for i, p in enumerate(data['areaTree'][0]['children']):
        if p['name'] == province:
            break
    today = list()
    total = list()
    
    for city in p['children']:
        city_today = city['today']
        city_today['city'] = city['name']
        city_today['province'] = province
        today.append(city_today)
        
    for city in p['children']:
        city_total = city['total']
        city_total['city'] = city['name']
        city_total['province'] = province
        total.append(city_total)
    
    return pd.DataFrame(today), pd.DataFrame(total)


# In[12]:


zhejiang_today, zhejiang_total = get_city_last_info(last_data, '浙江')
anhui_today, anhui_total = get_city_last_info(last_data, '安徽')
hubei_today, hubei_total = get_city_last_info(last_data, '湖北')
shanghai_today, shanghai_total = get_city_last_info(last_data, '上海')


# In[13]:


today = pd.concat((hubei_today, anhui_today, zhejiang_today, shanghai_today), axis=0)
today[today['province'] == '湖北']


# In[125]:


anhui_total


# # 2. data visualization
# ---
# 
# 
# ## 2.1 pyecharts
# ### 2.1.1 line chart

# In[19]:


from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.charts import Line


# In[20]:


hist_data = get_hist_data()
china_day_list = pd.DataFrame(hist_data['chinaDayList'])


# In[21]:


china_day_list.head()


# In[22]:


# 1. Construct line chart and use WALDENz theme
line = Line(init_opts=opts.InitOpts(theme=ThemeType.WALDEN))

# 2. add x axis data
line.add_xaxis(china_day_list['date'])

# 3. add y axis data
line.add_yaxis('nowConfirm', china_day_list['nowConfirm'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False,
              linestyle_opts=opts.LineStyleOpts(width=2))
line.add_yaxis('confirm', china_day_list['confirm'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False, 
               linestyle_opts=opts.LineStyleOpts(width=2))
line.add_yaxis('suspect', china_day_list['suspect'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False, 
               linestyle_opts=opts.LineStyleOpts(width=2))

# 4. Global options
line.set_global_opts(
            title_opts=opts.TitleOpts(title="Country nowConfirm/Suspect/Confirm trend", subtitle="Unit: Case"),
            yaxis_opts=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(width=2, color="white")),
                                     splitline_opts=opts.SplitLineOpts(is_show=True)
                                    ),
            xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45, font_family='Microsoft YaHei')),
            legend_opts=opts.LegendOpts(pos_right='right', pos_top='20%', legend_icon='circle', item_width=8),
            tooltip_opts=opts.TooltipOpts(is_show=True, trigger='axis', background_color='white', border_color='black', axis_pointer_type='line', 
                                          border_width=1,textstyle_opts=opts.TextStyleOpts(font_size=14, color='black')))
line.render_notebook()


# In[23]:


line = Line(init_opts=opts.InitOpts(theme=ThemeType.WALDEN))

line.add_xaxis(china_day_list['date'])
line.add_yaxis('heal', china_day_list['heal'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False,
              linestyle_opts=opts.LineStyleOpts(width=2))
line.add_yaxis('dead', china_day_list['dead'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False, 
               linestyle_opts=opts.LineStyleOpts(width=2))
line.set_global_opts(
            title_opts=opts.TitleOpts(title="Country heal/dead trend", subtitle="Unit: Case"),
            yaxis_opts=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(width=2, color="white")),
                                     splitline_opts=opts.SplitLineOpts(is_show=True)
                                    ),
            xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45, font_family='Microsoft YaHei')),
            legend_opts=opts.LegendOpts(pos_right='right', pos_top='20%', legend_icon='circle', item_width=8),
            tooltip_opts=opts.TooltipOpts(is_show=True, trigger='axis', background_color='white', border_color='black', axis_pointer_type='line', 
                                          border_width=1,textstyle_opts=opts.TextStyleOpts(font_size=14, color='black')))
line.render_notebook()


# In[24]:


china_day_add_list = pd.DataFrame(hist_data['chinaDayAddList'])

line = Line(init_opts=opts.InitOpts(theme=ThemeType.WALDEN))

line.add_xaxis(china_day_add_list['date'])
line.add_yaxis('NewHeal', china_day_add_list['heal'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False,
              linestyle_opts=opts.LineStyleOpts(width=2))
line.add_yaxis('NewConfirm', china_day_add_list['confirm'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False, 
               linestyle_opts=opts.LineStyleOpts(width=2))
line.add_yaxis('NewDead', china_day_add_list['dead'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False, 
               linestyle_opts=opts.LineStyleOpts(width=2))
line.add_yaxis('NewSuspect', china_day_add_list['suspect'], label_opts=opts.LabelOpts(is_show=False), is_symbol_show=False, 
               linestyle_opts=opts.LineStyleOpts(width=2))
line.set_global_opts(
            title_opts=opts.TitleOpts(title="Country Newconfirm/Newsuspect/Newheal/Newdead", subtitle="Unit: Case"),
            yaxis_opts=opts.AxisOpts(axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(width=2, color="white")),
                                     splitline_opts=opts.SplitLineOpts(is_show=True)
                                    ),
            xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45, font_family='Microsoft YaHei')),
            legend_opts=opts.LegendOpts(pos_right='right', pos_top='20%', legend_icon='circle', item_width=8),
            tooltip_opts=opts.TooltipOpts(is_show=True, trigger='axis', background_color='white', border_color='black', axis_pointer_type='line', 
                                          border_width=1,textstyle_opts=opts.TextStyleOpts(font_size=14, color='black')))
line.render_notebook()


# ### 2.1.2 Map

# In[32]:


from pyecharts import options as opts
from pyecharts.charts import Map


# In[33]:


confirms = list()
for province in last_data['areaTree'][0]['children']:
    province_name = province['name']
    province_data = province['total']['confirm'] - province['total']['heal'] - province['total']['dead']
    confirms.append([province_name, province_data])


# In[34]:


map_ = Map()
map_.add("Region", confirms, "china", is_map_symbol_show=False)
map_.set_global_opts(title_opts=opts.TitleOpts(title="Currently confirmed"), 
                     visualmap_opts=opts.VisualMapOpts(max_=40000, is_piecewise=True,
                                                       pieces=[
                                                           {"min": 10000},
                                                           {"min": 3000, "max": 10000}, 
                                                           {"min": 1500, "max": 3000},
                                                           {"min": 800, "max": 1500},
                                                           {"min": 400, "max": 800},
                                                           {"min": 200, "max": 400},
                                                           {"min": 100, "max": 200},
                                                           {"min": 60, "max": 100},
                                                           {"min": 30, "max": 60},
                                                           {"min": 5, "max": 30},
                                                           {"max": 5}
                                                          ],
#                                                        range_color
                                                      ))
map_.render_notebook()


# In[35]:


nameMap = {
            'Singapore Rep.':'新加坡',
            'Dominican Rep.':'多米尼加',
            'Palestine':'巴勒斯坦',
            'Bahamas':'巴哈马',
            'Timor-Leste':'东帝汶',
           'Afghanistan':'阿富汗',
           'Guinea-Bissau':'几内亚比绍',
           "Côte d'Ivoire":'科特迪瓦',
           'Siachen Glacier':'锡亚琴冰川',
           "Br. Indian Ocean Ter.":'英属印度洋领土',
           'Angola':'安哥拉',
           'Albania':'阿尔巴尼亚',
           'United Arab Emirates':'阿联酋',
           'Argentina':'阿根廷',
           'Armenia':'亚美尼亚',
           'French Southern and Antarctic Lands':'法属南半球和南极领地',
           'Australia':'澳大利亚',
           'Austria':'奥地利',
           'Azerbaijan':'阿塞拜疆',
           'Burundi':'布隆迪',
           'Belgium':'比利时',
           'Benin':'贝宁',
           'Burkina Faso':'布基纳法索',
           'Bangladesh':'孟加拉国',
           'Bulgaria':'保加利亚',
           'The Bahamas':'巴哈马',
           'Bosnia and Herz.':'波斯尼亚和黑塞哥维那',
           'Belarus':'白俄罗斯',
           'Belize':'伯利兹',
           'Bermuda':'百慕大',
           'Bolivia':'玻利维亚',
           'Brazil':'巴西',
           'Brunei':'文莱',
           'Bhutan':'不丹',
           'Botswana':'博茨瓦纳',
           'Central African Rep.':'中非',
           'Canada':'加拿大',
           'Switzerland':'瑞士',
           'Chile':'智利',
           'China':'中国',
           'Ivory Coast':'象牙海岸',
           'Cameroon':'喀麦隆',
           'Dem. Rep. Congo':'刚果民主共和国',
           'Congo':'刚果',
           'Colombia':'哥伦比亚',
           'Costa Rica':'哥斯达黎加',
           'Cuba':'古巴',
           'N. Cyprus':'北塞浦路斯',
           'Cyprus':'塞浦路斯',
           'Czech Rep.':'捷克',
           'Germany':'德国',
           'Djibouti':'吉布提',
           'Denmark':'丹麦',
           'Algeria':'阿尔及利亚',
           'Ecuador':'厄瓜多尔',
           'Egypt':'埃及',
           'Eritrea':'厄立特里亚',
           'Spain':'西班牙',
           'Estonia':'爱沙尼亚',
           'Ethiopia':'埃塞俄比亚',
           'Finland':'芬兰',
           'Fiji':'斐',
           'Falkland Islands':'福克兰群岛',
           'France':'法国',
           'Gabon':'加蓬',
           'United Kingdom':'英国',
           'Georgia':'格鲁吉亚',
           'Ghana':'加纳',
           'Guinea':'几内亚',
           'Gambia':'冈比亚',
           'Guinea Bissau':'几内亚比绍',
           'Eq. Guinea':'赤道几内亚',
           'Greece':'希腊',
           'Greenland':'格陵兰',
           'Guatemala':'危地马拉',
           'French Guiana':'法属圭亚那',
           'Guyana':'圭亚那',
           'Honduras':'洪都拉斯',
           'Croatia':'克罗地亚',
           'Haiti':'海地',
           'Hungary':'匈牙利',
           'Indonesia':'印度尼西亚',
           'India':'印度',
           'Ireland':'爱尔兰',
           'Iran':'伊朗',
           'Iraq':'伊拉克',
           'Iceland':'冰岛',
           'Israel':'以色列',
           'Italy':'意大利',
           'Jamaica':'牙买加',
           'Jordan':'约旦',
           'Japan':'日本',
           'Kazakhstan':'哈萨克斯坦',
           'Kenya':'肯尼亚',
           'Kyrgyzstan':'吉尔吉斯斯坦',
           'Cambodia':'柬埔寨',
           'Korea':'韩国',
           'Kosovo':'科索沃',
           'Kuwait':'科威特',
           'Lao PDR':'老挝',
           'Lebanon':'黎巴嫩',
           'Liberia':'利比里亚',
           'Libya':'利比亚',
           'Sri Lanka':'斯里兰卡',
           'Lesotho':'莱索托',
           'Lithuania':'立陶宛',
           'Luxembourg':'卢森堡',
           'Latvia':'拉脱维亚',
           'Morocco':'摩洛哥',
           'Moldova':'摩尔多瓦',
           'Madagascar':'马达加斯加',
           'Mexico':'墨西哥',
           'Macedonia':'北马其顿',
           'Mali':'马里',
           'Myanmar':'缅甸',
           'Montenegro':'黑山',
           'Mongolia':'蒙古',
           'Mozambique':'莫桑比克',
           'Mauritania':'毛里塔尼亚',
           'Malawi':'马拉维',
           'Malaysia':'马来西亚',
           'Namibia':'纳米比亚',
           'New Caledonia':'新喀里多尼亚',
           'Niger':'尼日尔',
           'Nigeria':'尼日利亚',
           'Nicaragua':'尼加拉瓜',
           'Netherlands':'荷兰',
           'Norway':'挪威',
           'Nepal':'尼泊尔',
           'New Zealand':'新西兰',
           'Oman':'阿曼',
           'Pakistan':'巴基斯坦',
           'Panama':'巴拿马',
           'Peru':'秘鲁',
           'Philippines':'菲律宾',
           'Papua New Guinea':'巴布亚新几内亚',
           'Poland':'波兰',
           'Puerto Rico':'波多黎各',
           'Dem. Rep. Korea':'朝鲜',
           'Portugal':'葡萄牙',
           'Paraguay':'巴拉圭',
           'Qatar':'卡塔尔',
           'Romania':'罗马尼亚',
           'Russia':'俄罗斯',
           'Rwanda':'卢旺达',
           'W. Sahara':'西撒哈拉',
           'Saudi Arabia':'沙特阿拉伯',
           'Sudan':'苏丹',
           'S. Sudan':'南苏丹',
           'Senegal':'塞内加尔',
           'Solomon Is.':'所罗门群岛',
           'Sierra Leone':'塞拉利昂',
           'El Salvador':'萨尔瓦多',
           'Somaliland':'索马里兰',
           'Somalia':'索马里',
           'Serbia':'塞尔维亚',
           'Suriname':'苏里南',
           'Slovakia':'斯洛伐克',
           'Slovenia':'斯洛文尼亚',
           'Sweden':'瑞典',
           'Swaziland':'斯威士兰',
           'Syria':'叙利亚',
           'Chad':'乍得',
           'Togo':'多哥',
           'Thailand':'泰国',
           'Tajikistan':'塔吉克斯坦',
           'Turkmenistan':'土库曼斯坦',
           'East Timor':'东帝汶',
           'Trinidad and Tobago':'特里尼达和多巴哥',
           'Tunisia':'突尼斯',
           'Turkey':'土耳其',
           'Tanzania':'坦桑尼亚',
           'Uganda':'乌干达',
           'Ukraine':'乌克兰',
           'Uruguay':'乌拉圭',
           'United States':'美国',
           'Uzbekistan':'乌兹别克斯坦',
           'Venezuela':'委内瑞拉',
           'Vietnam':'越南',
           'Vanuatu':'瓦努阿图',
           'West Bank':'西岸',
           'Yemen':'也门',
           'South Africa':'南非',
           'Zambia':'赞比亚',
           'Zimbabwe':'津巴布韦'
        }


# In[36]:


name_map_ = {value: key for key, value in nameMap.items()}
name_map_['日本本土'] = 'Japan'


# In[37]:


world_confirms = list()
last_data = get_last_data()
foreign_data = get_foreign_data()
for conuntry in last_data['areaTree']:
    try:
        conuntry_name = name_map_[conuntry['name']]
        conuntry_data = conuntry['total']['confirm'] - conuntry['total']['heal'] - conuntry['total']['dead']
        world_confirms.append([conuntry_name, conuntry_data])
    except Exception as e:
        print(conuntry['name'])

for country in foreign_data['foreignList']:
    try:
        country_name = name_map_[country['name']]
        country_data = country['confirm']
        world_confirms.append([country_name, country_data])
    except Exception as e:
        print(country['name'])


# In[38]:


map_ = Map()
map_.add("Country", world_confirms, "world", is_map_symbol_show=False)
map_.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
map_.set_global_opts(title_opts=opts.TitleOpts(title="Currently confirmed around the world"), 
                     visualmap_opts=opts.VisualMapOpts(max_=40000, is_piecewise=True,
                                                       pieces=[
                                                           {"min":100000},
                                                           {"min": 10000, "max":100000},
                                                           {"min": 3000, "max": 10000}, 
                                                           {"min": 1500, "max": 3000},
                                                           {"min": 800, "max": 1500},
                                                           {"min": 400, "max": 800},
                                                           {"min": 200, "max": 400},
                                                           {"min": 100, "max": 200},
                                                           {"min": 60, "max": 100},
                                                           {"min": 30, "max": 60},
                                                           {"min": 5, "max": 30},
                                                           {"max": 5}
                                                          ],
#                                                        range_color
                                                      ))
map_.render_notebook()


# # 3. Conclusion
# 
# * We use information technology to get access to some data about new corona virus and visualize it in Jupyter Notebook.
# * Using Jupyter Notebook can help us wirte codes, formulas and charts, and combine them!
# * IT helps us keep informed of the current situation and take timely measures to defeat it. It is beneficial to our society.
# * We hope everyone can make good use of it and finally overcome the difficulty nowadays! Thank you!
