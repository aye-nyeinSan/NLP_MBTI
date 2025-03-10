{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.upload()  # Manually upload kaggle.json\n"
      ],
      "metadata": {
        "id": "iNYrwDbGhUWs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!mkdir -p ~/.kaggle\n",
        "!mv kaggle.json ~/.kaggle/\n",
        "!chmod 600 ~/.kaggle/kaggle.json  # Set correct permissions\n"
      ],
      "metadata": {
        "id": "bZkZlE3yhrwh"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!kaggle datasets download -d zeyadkhalid/mbti-personality-types-500-dataset\n"
      ],
      "metadata": {
        "id": "uPxCItJmhva4",
        "outputId": "04a618aa-858c-4b4a-ee19-b2e8612fb37c",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset URL: https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset\n",
            "License(s): CC0-1.0\n",
            "Downloading mbti-personality-types-500-dataset.zip to /content\n",
            " 90% 111M/123M [00:00<00:00, 160MB/s] \n",
            "100% 123M/123M [00:00<00:00, 152MB/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!unzip mbti-personality-types-500-dataset.zip -d /content/my_dataset/\n"
      ],
      "metadata": {
        "id": "U21Flw22h9ja",
        "outputId": "0fb578f0-d9b9-405e-812e-1c17ec605ed3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Archive:  mbti-personality-types-500-dataset.zip\n",
            "  inflating: /content/my_dataset/MBTI 500.csv  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "df = pd.read_csv('/content/my_dataset/MBTI 500.csv')\n",
        "df.head()"
      ],
      "metadata": {
        "id": "_n7DxROsiQvu",
        "outputId": "081c6b05-418f-4ba5-f6e6-0020efc1aaea",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        }
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                               posts  type\n",
              "0  know intj tool use interaction people excuse a...  INTJ\n",
              "1  rap music ehh opp yeah know valid well know fa...  INTJ\n",
              "2  preferably p hd low except wew lad video p min...  INTJ\n",
              "3  drink like wish could drink red wine give head...  INTJ\n",
              "4  space program ah bad deal meing freelance max ...  INTJ"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-921e92b2-ab0b-4da3-8a15-9587354c1d0d\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>posts</th>\n",
              "      <th>type</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>know intj tool use interaction people excuse a...</td>\n",
              "      <td>INTJ</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>rap music ehh opp yeah know valid well know fa...</td>\n",
              "      <td>INTJ</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>preferably p hd low except wew lad video p min...</td>\n",
              "      <td>INTJ</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>drink like wish could drink red wine give head...</td>\n",
              "      <td>INTJ</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>space program ah bad deal meing freelance max ...</td>\n",
              "      <td>INTJ</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-921e92b2-ab0b-4da3-8a15-9587354c1d0d')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-921e92b2-ab0b-4da3-8a15-9587354c1d0d button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-921e92b2-ab0b-4da3-8a15-9587354c1d0d');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-9cd3371c-5896-4680-a307-79a9cdc034e1\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-9cd3371c-5896-4680-a307-79a9cdc034e1')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-9cd3371c-5896-4680-a307-79a9cdc034e1 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df"
            }
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile preprocessing.py\n",
        " # imports\n",
        "# type: ignore\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import re\n",
        "import nltk\n",
        "from nltk.tokenize import word_tokenize, sent_tokenize\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import PorterStemmer, WordNetLemmatizer\n",
        "\n",
        "nltk.download('punkt')\n",
        "nltk.download('stopwords')\n",
        "nltk.download('wordnet')\n",
        "\n",
        "\n",
        "data = pd.read_csv('data/MBTI500.csv')\n",
        "data = data.sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "\n",
        "def clean_text(text):\n",
        "    if pd.isna(text):  # Ensure text is not NaN\n",
        "        return \"\"\n",
        "\n",
        "    text = text.lower()\n",
        "\n",
        "    # Remove emoticons and emojis\n",
        "    text = re.sub(r'[^\\x00-\\x7F]+', '', text)\n",
        "\n",
        "    text = re.sub(r'[^a-z\\s]', '', text)\n",
        "\n",
        "    # Tokenize and remove stopwords\n",
        "    stop_words = set(stopwords.words('english'))\n",
        "    text = ' '.join(word for word in word_tokenize(text) if word not in stop_words)\n",
        "\n",
        "    # Lemmatization\n",
        "    lemmatizer = WordNetLemmatizer()\n",
        "    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())\n",
        "\n",
        "    # Stemming\n",
        "    stemmer = PorterStemmer()\n",
        "    text = ' '.join(stemmer.stem(word) for word in text.split())\n",
        "\n",
        "    return text\n",
        "\n",
        "data['posts'] = data['posts'].apply(clean_text)\n",
        "\n",
        "data['type'] = data['type'].str[0].fillna('I')\n",
        "\n",
        "data.to_csv('data/cleaned.csv', index=False)"
      ],
      "metadata": {
        "id": "AkMgVr3djXiv",
        "outputId": "c396959a-9f21-4f85-f78d-3ab0843548a7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing preprocessing.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git status\n"
      ],
      "metadata": {
        "id": "pESOY2KJjrSz",
        "outputId": "ccfde126-f05c-42d3-ca0c-540769c895e6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "On branch master\n",
            "Untracked files:\n",
            "  (use \"git add <file>...\" to include in what will be committed)\n",
            "\t\u001b[31msample_data/\u001b[m\n",
            "\n",
            "nothing added to commit but untracked files present (use \"git add\" to track)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "QHEKPL4ekkY9",
        "outputId": "b26a22a1-a7ee-4f91-c0a0-064af6288a84",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "On branch master\n",
            "Untracked files:\n",
            "  (use \"git add <file>...\" to include in what will be committed)\n",
            "\t\u001b[31m.config/\u001b[m\n",
            "\t\u001b[31mmbti-personality-types-500-dataset.zip\u001b[m\n",
            "\t\u001b[31mmy_dataset/\u001b[m\n",
            "\t\u001b[31msample_data/\u001b[m\n",
            "\n",
            "nothing added to commit but untracked files present (use \"git add\" to track)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git add preprocessing.py\n",
        "!git commit -m \"ADD: preprocessing\"\n",
        "!git push --set-upstream origin master"
      ],
      "metadata": {
        "id": "87NDAViVjrVa",
        "outputId": "13eaf773-a2f9-4117-a042-c9433811f1d2",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "On branch master\n",
            "Untracked files:\n",
            "  (use \"git add <file>...\" to include in what will be committed)\n",
            "\t\u001b[31m.config/\u001b[m\n",
            "\t\u001b[31mmbti-personality-types-500-dataset.zip\u001b[m\n",
            "\t\u001b[31mmy_dataset/\u001b[m\n",
            "\t\u001b[31msample_data/\u001b[m\n",
            "\n",
            "nothing added to commit but untracked files present (use \"git add\" to track)\n",
            "fatal: could not read Username for 'https://github.com': No such device or address\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.13.2"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}