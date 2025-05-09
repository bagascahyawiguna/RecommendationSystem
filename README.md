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

Dataset MovieLens terdiri dari empat file utama:

* **movies.csv**: berisi 9.742 film unik beserta genre.
* **ratings.csv**: berisi 1.000.209 entri rating dari 610 pengguna.
* **tags.csv**: berisi 368.3 tag dari pengguna.
* **links.csv**: berisi 9.742 tautan IMDb dan TMDb.

Berikut adalah beberapa variabel utama dalam dataset:

* **movieId:** ID unik film.
* **userId:** ID unik pengguna.
* **rating:** Skor rating yang diberikan (1-5).
* **tag:** Label/tag yang diberikan oleh pengguna terhadap film tertentu.
* **title:** Judul film beserta tahun rilis.
* **genres:** Genre film, dipisahkan oleh tanda '|'.

### Visualisasi Data

1. Distribusi rating menunjukkan adanya bias positif dengan mayoritas rating berkisar antara 3 hingga 5.
2. Genre yang paling banyak muncul adalah Drama dan Comedy.
3. Tag paling populer adalah "In Netflix queue", "atmospheric", dan "thought-provoking".

## Data Preparation

Langkah-langkah data preparation yang dilakukan adalah:

1. **Mengatasi Missing Values:** Kolom tmdbId memiliki 8 nilai kosong yang tidak digunakan dalam model.
2. **Mengatasi Duplikasi:** Tidak ditemukan duplikasi pada dataset.
3. **Filtering Data:** Menghapus film yang memiliki kurang dari 10 rating dan pengguna yang memberikan kurang dari 10 rating untuk mengurangi sparsity.
4. **Mempersiapkan Data untuk Content-Based Filtering:** Menggabungkan genre dan tag menjadi kolom 'content' untuk digunakan dalam TF-IDF Vectorization.
5. **Mempersiapkan Data untuk Collaborative Filtering:** Melakukan encoding pada userId dan movieId untuk digunakan dalam model neural network.

## Modeling and Result

Pendekatan yang digunakan dalam proyek ini adalah:

1. **Content-Based Filtering:**

   * Menggunakan TF-IDF Vectorizer untuk merepresentasikan konten film berdasarkan genre dan tag.
   * Menggunakan Cosine Similarity untuk mengukur kemiripan antar film.
   * Output berupa top-10 rekomendasi film berdasarkan kemiripan konten.

2. **Collaborative Filtering:**

   * Menggunakan neural network dengan embedding layer untuk merepresentasikan userId dan movieId.
   * Melakukan prediksi rating menggunakan sigmoid activation function.
   * Output berupa top-10 rekomendasi film berdasarkan preferensi pengguna lain yang mirip.

### Kelebihan dan Kekurangan Pendekatan:

* **Content-Based Filtering:**

  * Kelebihan: Tidak membutuhkan data rating pengguna lain.
  * Kekurangan: Terbatas pada konten yang diketahui; tidak bisa merekomendasikan film yang tidak mirip dengan preferensi sebelumnya.

* **Collaborative Filtering:**

  * Kelebihan: Dapat merekomendasikan film baru yang tidak pernah ditonton pengguna.
  * Kekurangan: Membutuhkan data rating pengguna lain dan dapat terpengaruh oleh sparsity data.

## Evaluation

Metrik evaluasi yang digunakan adalah:

1. **Content-Based Filtering:** Tidak dilakukan evaluasi berbasis rating karena model tidak melakukan prediksi rating. Evaluasi dilakukan berdasarkan kemiripan konten (Cosine Similarity).
2. **Collaborative Filtering:**

   * Menggunakan metrik Root Mean Squared Error (RMSE) untuk mengevaluasi perbedaan antara rating asli dan prediksi rating.
   * RMSE pada data validation berkisar di 0.20, yang menunjukkan bahwa model dapat memprediksi rating dengan akurasi yang cukup baik pada skala 0–1.

### Formula RMSE:

RMSE = $\sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$

RMSE dihitung untuk memastikan model tidak terlalu overfitting dan tetap dapat melakukan generalisasi pada data baru.

**--- Ini adalah bagian akhir laporan ---**
