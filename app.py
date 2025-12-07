import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Настройка страницы Streamlit
st.set_page_config(layout="wide", page_title="Система рекомендаций одежды")

# Загрузка данных
@st.cache_data  # кэшируем данные для ускорения повторных запусков
def load_data():
    # Загрузка CSV с информацией о товарах
    styles = pd.read_csv("styles.csv", on_bad_lines='skip')
    # Загрузка CSV с изображениями и их ссылками
    images = pd.read_csv("images.csv")
    # Загрузка предвычисленных фичей ResNet50
    features = joblib.load("resnet50_features.pkl")
    # Загрузка списка файлов изображений
    filenames = joblib.load("resnet50_files.pkl")

    # Убираем строки без id
    styles = styles[styles['id'].notna()]
    styles['id'] = styles['id'].astype(int)  # приводим id к int
    styles['image_path'] = styles['id'].astype(str) + ".jpg"  # формируем путь к файлу

    # Оставляем только те товары, у которых есть изображение
    styles = styles[styles['image_path'].isin(filenames)]

    # Сопоставляем локальные файлы с внешними ссылками
    image_map = dict(zip(images['filename'], images['link']))
    styles['image_url'] = styles['image_path'].map(image_map)

    return styles, features, filenames

# Загружаем данные
styles, features, filenames = load_data()

# Инициализация корзины
if "cart" not in st.session_state:
    st.session_state.cart = []  # создаём пустую корзину при первом запуске

# Добавление товара в корзину
def add_to_cart(image_url, product_name, category, color, season, usage, year):
    st.session_state.cart.append({
        "image_url": image_url,       # URL изображения
        "product_name": product_name, # Название товара
        "category": category,         # Категория
        "color": color,               # Цвет
        "season": season,             # Сезон
        "usage": usage,               # Назначение
        "year": year                  # Год
    })

# Удаление товара из корзины
def remove_from_cart(image_url):
    # Фильтруем корзину, исключая выбранный товар
    st.session_state.cart = [item for item in st.session_state.cart if item["image_url"] != image_url]

# Отображение корзины
def show_cart():
    if st.session_state.cart:
        st.markdown("## 🛍️ Ваша корзина")
        st.markdown("---")
        for item in st.session_state.cart:
            with st.container():
                col1, col2 = st.columns([1, 3])  # создаём 2 колонки для изображения и описания
                with col1:
                    st.image(item["image_url"], use_container_width=False, width=250)  # показываем картинку
                with col2:
                    # Выводим информацию о товаре
                    content = f"""
                    <div style='display: flex; flex-direction: column; justify-content: center; height: 100%;'>
                        <p><strong>{item['product_name']}</strong></p>
                        <p>Категория: {category_dict.get(item['category'], item['category'])}</p>
                        <p>Цвет: {color_dict.get(item['color'], item['color'])}</p>
                        <p>Сезон: {season_dict.get(item['season'], item['season'])}</p>
                        <p>Назначение: {usage_dict.get(item['usage'], item['usage'])}</p>
                        <p>Год: {item['year']}</p>
                    </div>
                    """
                    st.markdown(content, unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.markdown("## 🛍️ Ваша корзина пуста")  # вывод при пустой корзине

# Словари для перевода
# Перевод категорий, подкатегорий, полов, сезонов, назначения и цветов на русский
category_dict = {'Apparel': 'Одежда', 'Accessories': 'Аксессуары', 'Footwear': 'Обувь', 'Personal Care': 'Уход за собой', 'Free Items': 'Подарочные товары'}
sub_category_dict = {'Flip Flops': 'Шлёпанцы', 'Sandal': 'Сандалии', 'Skin Care': 'Уход за кожей', 'Saree': 'Сари', 'Free Gifts': 'Подарочные товары', 'Ties': 'Галстуки', 'Accessories': 'Аксессуары', 'Shoe Accessories': 'Аксессуары для обуви', 'Lips': 'Помада', 'Apparel Set': 'Комплект одежды', 'Scarves': 'Шарфы', 'Innerwear': 'Нижнее бельё', 'Topwear': 'Верхняя одежда', 'Bottomwear': 'Низ одежды', 'Loungewear and Nightwear': 'Одежда для отдыха и ночная одежда', 'Dress': 'Платья', 'Fragrance': 'Парфюмерия', 'Makeup': 'Макияж', 'Nails': 'Уход за ногтями', 'Eyewear': 'Очки', 'Watches': 'Часы', 'Bags': 'Сумки', 'Jewellery': 'Ювелирные изделия', 'Belts': 'Ремни', 'Wallets': 'Кошельки', 'Socks': 'Носки', 'Cufflinks': 'Запонки', 'Headwear': 'Головные уборы'}
gender_dict = {'Men': 'Мужчины', 'Women': 'Женщины', 'Unisex': 'Унисекс', 'Boys': 'Мальчики', 'Girls': 'Девочки'}
season_dict = {'Summer': 'Лето', 'Winter': 'Зима', 'Fall': 'Осень', 'Spring': 'Весна'}
usage_dict = {'Casual': 'Повседневный стиль', 'Sports': 'Спортивный стиль', 'Ethnic': 'Этнический стиль', 'Formal': 'Официальный стиль', 'Travel': 'Путешествия'}
color_dict = {'White': 'Белый', 'Grey': 'Серый', 'Black': 'Чёрный', 'Silver': 'Серебристый', 'Blue': 'Синий', 'Brown': 'Коричневый', 'Green': 'Зелёный', 'Red': 'Красный', 'Lavender': 'Лаванда', 'Beige': 'Бежевый', 'Orange': 'Оранжевый', 'Gold': 'Золотой', 'Cream': 'Кремовый', 'Pink': 'Розовый', 'Navy Blue': 'Тёмно-синий', 'Peach': 'Персиковый', 'Yellow': 'Жёлтый', 'Steel': 'Стальной', 'Mustard': 'Горчичный', 'Maroon': 'Тёмно-вишнёвый', 'Teal': 'Тёмно-бирюзовый', 'Off White': 'Не совсем белый', 'Purple': 'Фиолетовый', 'Skin': 'Кожа', 'Turquoise Blue': 'Бирюзовый', 'Copper': 'Медный', 'Charcoal': 'Угольный', 'Olive': 'Оливковый', 'Magenta': 'Пурпурный', 'Rust': 'Ржавый', 'Grey Melange': 'Серый меланж', 'Multi': 'Мультицветный', 'Fluorescent Green': 'Флуоресцентный зелёный'}

# Фильтры
st.sidebar.header("🔍 Фильтры")

# Фильтр по полу
gender = st.sidebar.multiselect("Пол", options=[gender_dict.get(g, g) for g in styles['gender'].dropna().unique()])
# Фильтр по категории
category = st.sidebar.multiselect("Категория", options=[category_dict.get(c, c) for c in styles['masterCategory'].dropna().unique()])

# Фильтр подкатегорий в зависимости от выбранной категории
sub_categories_filtered = []
if category:
    selected_category_english = [k for k, v in category_dict.items() if v in category]
    sub_categories_filtered = styles[styles['masterCategory'].isin(selected_category_english)]['subCategory'].unique()

# Перевод подкатегорий на русский
sub_category = st.sidebar.multiselect("Подкатегория", options=[sub_category_dict.get(s, s) for s in sub_categories_filtered])

# Для одежды, обуви и аксессуаров показываем фильтры по сезону и назначению
show_season_usage = any(cat in ['Одежда', 'Аксессуары', 'Обувь'] for cat in category)
season = usage = []
if show_season_usage:
    season = st.sidebar.multiselect("Сезон", options=[season_dict.get(s, s) for s in styles['season'].dropna().unique()])
    usage = st.sidebar.multiselect("Назначение", options=[usage_dict.get(u, u) for u in styles['usage'].dropna().unique()])

# Фильтр по цвету
color_filtered = []
if any(cat in ['Одежда', 'Аксессуары', 'Обувь'] for cat in category):
    color_filtered = styles['baseColour'].dropna().unique()
color_translated = [color_dict.get(c, c) for c in color_filtered]
color = []
if category and any(cat in ['Одежда', 'Аксессуары', 'Обувь'] for cat in category):
    color = st.sidebar.multiselect("Цвет", options=color_translated)

# Применение фильтров
filtered_styles = styles.copy()
if gender:
    filtered_styles = filtered_styles[filtered_styles['gender'].isin([k for k, v in gender_dict.items() if v in gender])]
if category:
    filtered_styles = filtered_styles[filtered_styles['masterCategory'].isin([k for k, v in category_dict.items() if v in category])]
if sub_category:
    filtered_styles = filtered_styles[filtered_styles['subCategory'].isin([k for k, v in sub_category_dict.items() if v in sub_category])]
if show_season_usage:
    if season:
        filtered_styles = filtered_styles[filtered_styles['season'].isin([k for k, v in season_dict.items() if v in season])]
    if usage:
        filtered_styles = filtered_styles[filtered_styles['usage'].isin([k for k, v in usage_dict.items() if v in usage])]
if color:
    color_english = [k for k, v in color_dict.items() if v in color]
    filtered_styles = filtered_styles[filtered_styles['baseColour'].isin(color_english)]

st.title("👗 Система рекомендаций одежды")

# Вывод товаров
if "selected_image_id" not in st.session_state:
    st.session_state["selected_image_id"] = None

selected_image_index = None
cols = st.columns(5)

for idx, (i, row) in enumerate(filtered_styles.head(10).iterrows()):
    with cols[idx % 5]:
        is_selected = st.session_state["selected_image_id"] == row["image_path"]
        if st.checkbox("Выбрать", key=f"select_{i}", value=is_selected):
            st.session_state["selected_image_id"] = row["image_path"] if not is_selected else None

        # Отображение изображения и названия товара
        st.markdown(
            f"""
            <div style='padding: 0; margin: 0; border: none; box-shadow: none;'>
                <div style='height: 300px; display: flex; align-items: center; justify-content: center; overflow: hidden;'>
                    <img src="{row["image_url"]}" style="height: 100%; object-fit: cover;">
                </div>
                <div style='text-align: center; color: #666666; font-weight: normal; font-size: 14px; height: 3em; overflow: hidden; text-overflow: ellipsis; margin-top: 5px;'>
                    {row["productDisplayName"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Кнопка "Добавить в корзину"
        if st.button("Добавить в корзину", key=f"add_{i}"):
            add_to_cart(
                row["image_url"],
                row["productDisplayName"],
                row["masterCategory"],
                row["baseColour"],
                row.get("season", ""),
                row.get("usage", ""),
                row.get("year", "")
            )

# Похожие товары
if st.session_state["selected_image_id"]:
    st.markdown("---")
    st.subheader("🧠 Похожие товары")
    selected_image_index = filenames.index(st.session_state["selected_image_id"])
    similarities = cosine_similarity([features[selected_image_index]], features)[0]  # косинусная схожесть
    top_indices = similarities.argsort()[-11:-1][::-1]  # 5 самых похожих (исключая сам запрос)
    # Берём товары, изображения которых соответствуют индексам топ-5 похожих
    recs = styles[styles['image_path'].isin([filenames[i] for i in top_indices])]
    # Создаём 5 колонок для отображения рекомендаций
    rec_cols = st.columns(5)
    # Итерация по выбранным рекомендациям
    for idx, (i, row) in enumerate(recs.iterrows()):
        with rec_cols[idx % 5]: # распределяем товары по колонкам, по циклу
            st.image(row["image_url"], caption=row["productDisplayName"], use_container_width=True) # показываем изображение с подписью
            # Кнопка "Добавить в корзину" для каждой рекомендации
            if st.button("Добавить в корзину", key=f"add_rec_{i}"):
                # Добавляем товар в корзину через функцию add_to_cart
                add_to_cart(
                    row["image_url"], row["productDisplayName"], row["masterCategory"],
                    row["baseColour"], row.get("season", ""), row.get("usage", ""), row.get("year", "")
                )

# Боковая панель с корзиной
st.sidebar.header("🛒 Корзина")
if st.session_state.cart:
    st.sidebar.write(f"Товары: {len(st.session_state.cart)}")
    st.sidebar.button("Перейти в корзину", on_click=show_cart)
else:
    st.sidebar.write("Корзина пуста.")