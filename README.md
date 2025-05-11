# Laporan Proyek Machine Learning - Sistem Rekomendasi Film GroupLens MovieLens - Bagas Cahyawiguna

## Project Overview

Sistem rekomendasi merupakan salah satu aplikasi machine learning yang sangat berguna dalam menyaring informasi dan membantu pengguna untuk menemukan item yang relevan berdasarkan preferensi mereka. Dalam proyek ini, sistem rekomendasi dibangun menggunakan dataset MovieLens dari GroupLens, sebuah dataset populer yang berisi informasi rating film, tag, genre, dan tautan ke IMDb dan TMDb.

Dataset ini terdiri dari beberapa tabel utama, yaitu:

* **movies.csv**: berisi informasi judul film dan genre.
* **ratings.csv**: berisi informasi penilaian film oleh pengguna.
* **tags.csv**: berisi tag atau label yang diberikan oleh pengguna terhadap film tertentu.
* **links.csv**: berisi tautan ke IMDb dan TMDb untuk setiap film.

Proyek ini menggunakan dua pendekatan utama:

1. Content-Based Filtering, yang merekomendasikan film berdasarkan kemiripan konten (genre dan tag).
2. Collaborative Filtering, yang merekomendasikan film berdasarkan pola rating pengguna lain yang mirip.

### Mengapa Proyek Ini Penting?

Dengan semakin banyaknya film baru yang dirilis setiap tahunnya, pengguna dapat merasa kewalahan dalam memilih film yang relevan dengan preferensi mereka. Sistem rekomendasi dapat membantu mengurangi informasi berlebih dan memberikan rekomendasi yang lebih terpersonalisasi.

### Referensi:

* Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.
* MovieLens Dataset: [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)

## Business Understanding

### Problem Statements:

1. Bagaimana cara merekomendasikan film berdasarkan kemiripan konten (genre dan tag)?
2. Bagaimana cara merekomendasikan film berdasarkan pola rating pengguna lain yang memiliki preferensi serupa?

### Goals:

1. Mengembangkan model Content-Based Filtering untuk memberikan rekomendasi film berdasarkan genre dan tag.
2. Mengembangkan model Collaborative Filtering untuk memberikan rekomendasi film berdasarkan perilaku pengguna lain yang mirip.

### Solution Statements:

* **Content-Based Filtering:** Menggunakan teknik TF-IDF untuk merepresentasikan konten film dan menghitung kemiripan menggunakan Cosine Similarity.
* **Collaborative Filtering:** Menggunakan model neural network sederhana dengan embedding layer untuk merekomendasikan film berdasarkan rating pengguna.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens 1M** dari GroupLens, yang dapat diakses melalui tautan berikut:
🔗 [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)

Dataset ini terdiri dari empat file utama, masing-masing memiliki struktur dan informasi sebagai berikut:

### 1. `movies.csv`

* **Ukuran data**: 9.742 baris × 3 kolom
* **Fitur**:

  * `movieId`: ID unik untuk setiap film.
  * `title`: Judul film beserta tahun rilis.
  * `genres`: Genre film yang dipisahkan dengan tanda pipe (`|`).
* **Kondisi data**:

  * Tidak ditemukan nilai kosong (missing values).
  * Tidak ditemukan duplikasi.

### 2. `ratings.csv`

* **Ukuran data**: 100.836 baris × 4 kolom
* **Fitur**:

  * `userId`: ID unik pengguna.
  * `movieId`: ID film yang diberi rating.
  * `rating`: Nilai rating dari pengguna (skala 0.5 – 5.0).
  * `timestamp`: Waktu pemberian rating dalam UNIX timestamp.
* **Kondisi data**:

  * Tidak ada missing value.
  * Tidak ditemukan duplikasi.
  * Distribusi rating menunjukkan bias positif, dengan sebagian besar nilai antara 3 hingga 5.

### 3. `tags.csv`

* **Ukuran data**: 3683 baris × 4 kolom
* **Fitur**:

  * `userId`: ID pengguna yang memberikan tag.
  * `movieId`: ID film yang diberi tag.
  * `tag`: Label atau deskripsi yang diberikan pengguna untuk film.
  * `timestamp`: Waktu pemberian tag.
* **Kondisi data**:

  * Beberapa nilai kosong ditemukan di kolom `tag`.
  * Tag yang sama bisa diberikan oleh pengguna berbeda untuk film yang sama (tidak dianggap duplikat).

### 4. `links.csv`

* **Ukuran data**: 9.742 baris × 3 kolom
* **Fitur**:

  * `movieId`: ID film.
  * `imdbId`: ID film di IMDb.
  * `tmdbId`: ID film di TMDb.
* **Kondisi data**:

  * Ditemukan 8 nilai kosong pada kolom `tmdbId`.
  * Tidak digunakan langsung dalam pemodelan, sehingga tidak dilakukan imputasi.

### Visualisasi Awal Data

* **Distribusi Rating**:

  * Skor rating didominasi nilai tinggi (3.0–5.0), menandakan adanya bias pengguna dalam memberikan rating.
* **Genre Populer**:

  * Genre terbanyak adalah *Drama*, *Comedy*, *Action*, dan *Thriller*.
* **Tag Populer**:

  * Tag yang sering muncul antara lain: `"In Netflix queue"`, `"atmospheric"`, `"thought-provoking"`.

## Data Preparation

Tahapan persiapan data dilakukan secara sistematis agar model yang dibangun dapat bekerja dengan optimal. Berikut adalah langkah-langkah yang dilakukan:

### 1. **Pemeriksaan dan Penanganan Missing Values**

* Pada dataset `links.csv`, ditemukan **8 nilai kosong pada kolom `tmdbId`**.
* Karena kolom ini **tidak digunakan langsung dalam pemodelan**, maka **tidak dilakukan imputasi** atau penghapusan baris.

### 2. **Pemeriksaan Duplikasi**

* Tidak ditemukan baris duplikat pada seluruh dataset (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`).
* Karena tidak dilakukan tindakan terhadap duplikat, bagian ini **diabaikan**.

### 3. **Penggabungan Dataset**

* Data dari `movies.csv`, `ratings.csv`, dan `tags.csv` **digabungkan berdasarkan `movieId`**.
* Tujuan penggabungan adalah untuk menyatukan informasi konten film (genre, tag) dan perilaku pengguna (rating) dalam satu tabel untuk keperluan content-based dan collaborative filtering.

### 4. **Filtering Data (Penting untuk Collaborative Filtering)**

* Dilakukan penyaringan data untuk mengurangi sparsity dan meningkatkan kualitas pembelajaran model:

  * **Hanya film dengan ≥10 rating yang disertakan.**
  * **Hanya pengguna dengan ≥10 rating yang disertakan.**
* Filtering dilakukan **sebelum proses pelatihan model Collaborative Filtering**, dan **sebelum pembentukan dataset akhir**.

### 5. **Pembuatan Fitur Konten untuk Content-Based Filtering**

* Genre dan tag digabung menjadi satu kolom `content`, lalu dilakukan:

  * **Text cleaning**: lowercase, penghapusan simbol, dsb.
  * **TF-IDF Vectorization**: untuk mengubah konten menjadi vektor numerik.
* TF-IDF dilakukan terhadap semua film, bukan hanya yang ada dalam subset training.

### 6. **Encoding ID untuk Collaborative Filtering**

* Kolom `userId` dan `movieId` diencoding menjadi indeks numerik menggunakan `StringLookup` dari TensorFlow.
* Proses ini dilakukan agar dapat digunakan dalam **Embedding Layer** pada model RecommenderNet.

### 7. **Split Data untuk Pelatihan dan Validasi**

* Dataset dibagi menjadi:

  * **Training set**: 80%
  * **Validation set**: 20%
* Pembagian dilakukan **secara acak**, tetapi hanya setelah proses encoding dan filtering selesai.

Dengan tahapan ini, data telah dipersiapkan secara lengkap untuk dua pendekatan rekomendasi:

* **Content-Based Filtering** → TF-IDF dari konten film.
* **Collaborative Filtering** → dataset terstruktur dengan pasangan `(userId, movieId)` dan target `rating`.

## Modeling and Result

Proyek ini mengimplementasikan dua pendekatan sistem rekomendasi: **Content-Based Filtering** dan **Collaborative Filtering** menggunakan neural network. Berikut penjelasan detail kedua pendekatan beserta contoh hasil top-N rekomendasinya:

### 1. Content-Based Filtering

#### Tujuan:

Memberikan rekomendasi film berdasarkan **kemiripan konten**, yaitu kombinasi **genre dan tag** dari masing-masing film.

#### Teknik dan Proses:

* **Fitur konten** dibuat dengan menggabungkan `genres` dan `tags` ke dalam satu kolom `content`.
* Dilakukan **TF-IDF Vectorization** menggunakan `TfidfVectorizer` dari Scikit-learn:

  ```python
  tfidf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf.fit_transform(movies_content['content'])
  ```
* Matriks **cosine similarity** dihitung antara semua film:

  ```python
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
  cosine_sim_df = pd.DataFrame(cosine_sim, index=movies_content['title'], columns=movies_content['title'])
  ```
* Fungsi `movie_recommendations()` dikembangkan untuk menghasilkan **Top-10 rekomendasi** berdasarkan judul film input.

#### Contoh Hasil Rekomendasi: "Toy Story (1995)"

| Peringkat | Judul Film                       | Genre                                  |
| --------- | -------------------------------- | -------------------------------------- |
| 1         | A Bug's Life (1998)              | Adventure, Animation, Children, Comedy |
| 2         | Toy Story 2 (1999)               | Adventure, Animation, Children, Comedy |
| 3         | Guardians of the Galaxy 2 (2017) | Action, Adventure, Sci-Fi              |
| 4         | Asterix and the Vikings (2006)   | Adventure, Animation, Children, Comedy |
| 5         | Shrek the Third (2007)           | Adventure, Animation, Children, Comedy |
| 6         | Monsters, Inc. (2001)            | Adventure, Animation, Children, Comedy |
| 7         | Turbo (2013)                     | Adventure, Animation, Children, Comedy |
| 8         | The Good Dinosaur (2015)         | Adventure, Animation, Children, Comedy |
| 9         | Wild, The (2006)                 | Adventure, Animation, Children, Comedy |
| 10        | Emperor's New Groove, The (2000) | Adventure, Animation, Children, Comedy |


### 2. Collaborative Filtering – RecommenderNet (TensorFlow)

#### Tujuan:

Mempersonalisasi rekomendasi berdasarkan **interaksi pengguna**, dengan mempelajari pola rating dari pengguna yang serupa.

#### Teknik dan Proses:

* **Encoding**:

  * `userId` dan `movieId` diencoding ke indeks numerik.
* **Data Split**:

  * Dataset diacak dan dibagi menjadi **80% data training** dan **20% data validasi**.
* **Normalisasi Rating**:

  * Rating dinormalisasi ke skala 0–1 sebelum digunakan sebagai target prediksi.

#### Arsitektur Model:

`RecommenderNet`, model kustom berbasis TensorFlow, memiliki struktur sebagai berikut:

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size=50):
        ...
```

* **Input**: pasangan (`user`, `movie`)
* **Output**: prediksi rating (0–1) menggunakan fungsi aktivasi `sigmoid`
* **Loss Function**: `BinaryCrossentropy`
* **Optimizer**: Adam
* **Epochs**: 20
* **Batch Size**: 64

#### Hasil Pelatihan Model:

* **RMSE Training** dan **Validation** menurun secara konsisten.
* **Tidak terjadi overfitting** yang signifikan.
* **RMSE pada data validasi \~0.20**, menunjukkan prediksi cukup akurat (ingat: skala rating telah dinormalisasi ke 0–1).

#### Contoh Hasil Rekomendasi untuk User ID: 237

**Film dengan rating tertinggi oleh user:**

* Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
* Shawshank Redemption, The (1994)
* Game, The (1997)
* Men in Black (a.k.a. MIB) (1997)
* One Flew Over the Cuckoo's Nest (1975)

**Top-10 rekomendasi berdasarkan prediksi model:**

| Peringkat | Judul Film                                                      |
| --------- | --------------------------------------------------------------- |
| 1         | City of Lost Children, The (Cité des enfants perdus, La) (1995) |
| 2         | Once Upon a Time in the West (C'era una volta il West) (1968)   |
| 3         | Godfather: Part II, The (1974)                                  |
| 4         | Raging Bull (1980)                                              |
| 5         | Rosemary's Baby (1968)                                          |
| 6         | Double Indemnity (1944)                                         |
| 7         | Guess Who's Coming to Dinner (1967)                             |
| 8         | Lost in Translation (2003)                                      |
| 9         | There Will Be Blood (2007)                                      |
| 10        | Toy Story 3 (2010)                                              |

## Evaluation

Evaluasi dilakukan untuk menilai efektivitas model dalam memberikan rekomendasi yang relevan bagi pengguna, serta untuk menghubungkan performa model terhadap tujuan bisnis dan problem statement yang telah ditetapkan.

### Evaluasi Content-Based Filtering

#### Metrik yang Digunakan: **Precision\@K berbasis Genre**

Karena pendekatan Content-Based Filtering tidak memprediksi rating eksplisit, maka evaluasi dilakukan dengan menggunakan **Precision\@10**, yaitu:

> Proporsi dari 10 film yang direkomendasikan yang memiliki **genre serupa** dengan film input.

Ini dilakukan dengan asumsi bahwa film yang memiliki genre yang tumpang tindih dianggap relevan oleh pengguna.

#### Hasil Evaluasi:

Untuk film input **"Toy Story (1995)"**, diperoleh hasil:

* **Precision\@10 = 1.00**, yang berarti seluruh 10 film yang direkomendasikan memiliki genre yang serupa, seperti Animation, Adventure, dan Comedy.

> Ini menunjukkan bahwa model **berhasil memberikan rekomendasi yang sangat relevan** secara konten.

### Evaluasi Collaborative Filtering

#### Metrik yang Digunakan: **Root Mean Squared Error (RMSE)**

##### Formula RMSE:

RMSE = $\sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$

RMSE dihitung untuk memastikan model tidak terlalu overfitting dan tetap dapat melakukan generalisasi pada data baru.

Model Collaborative Filtering (RecommenderNet) memprediksi rating numerik (setelah normalisasi ke skala 0–1). Maka digunakan metrik RMSE untuk mengukur akurasi prediksi.

* RMSE pada data validasi sekitar **0.20**
* Kurva learning menunjukkan **tidak terjadi overfitting**
* Gap antara training dan validation RMSE **sangat kecil**

> Ini berarti model mampu melakukan generalisasi dan memprediksi rating dengan cukup akurat.

### Keterkaitan dengan Business Understanding

| Pertanyaan Evaluasi                                               | Jawaban                                                                                                                    |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Apakah model menjawab problem statement?                          | Ya. Content-Based menjawab kemiripan konten; Collaborative menyasar pola perilaku pengguna.                              |
| Apakah model mencapai goals?                                      | Ya. Dua sistem rekomendasi berhasil dikembangkan dan diuji.                                                              |
| Apakah solusi yang diusulkan berdampak terhadap kebutuhan bisnis? | Ya. Sistem mampu memberikan rekomendasi yang relevan dan personal, membantu pengguna menjelajahi film sesuai preferensi. |

### Kesimpulan

* **Content-Based Filtering** terbukti akurat dalam merekomendasikan film yang mirip secara konten.
* **Collaborative Filtering** menunjukkan performa prediktif yang baik.
* Sistem ini dapat memberikan **nilai nyata bagi pengguna**, terutama dalam platform pencarian atau streaming film.

**--- Ini adalah bagian akhir laporan ---**
