

# Employee-salary-prediction-
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fb41786e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import Libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7b30d022",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv(\"Salary.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e6ccc7a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>Country</th>\n",
       "      <th>Education Level</th>\n",
       "      <th>Years of Experience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Germany</td>\n",
       "      <td>Master’s degree (M.A., M.S., M.Eng., MBA, etc.)</td>\n",
       "      <td>36</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>United Kingdom</td>\n",
       "      <td>Bachelor’s degree (B.A., B.S., B.Eng., etc.)</td>\n",
       "      <td>7</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Russian Federation</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Albania</td>\n",
       "      <td>Master’s degree (M.A., M.S., M.Eng., MBA, etc.)</td>\n",
       "      <td>7</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>United States</td>\n",
       "      <td>Bachelor’s degree (B.A., B.S., B.Eng., etc.)</td>\n",
       "      <td>15</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Country                                  Education Level  \\\n",
       "0             Germany  Master’s degree (M.A., M.S., M.Eng., MBA, etc.)   \n",
       "1      United Kingdom     Bachelor’s degree (B.A., B.S., B.Eng., etc.)   \n",
       "2  Russian Federation                                              NaN   \n",
       "3             Albania  Master’s degree (M.A., M.S., M.Eng., MBA, etc.)   \n",
       "4       United States     Bachelor’s degree (B.A., B.S., B.Eng., etc.)   \n",
       "\n",
       "  Years of Experience  Salary  \n",
       "0                  36     NaN  \n",
       "1                   7     NaN  \n",
       "2                   4     NaN  \n",
       "3                   7     NaN  \n",
       "4                  15     NaN  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Select and rename columns in one go\n",
    "df = df[[\"Country\", \"EdLevel\", \"YearsCode\", \"ConvertedComp\"]]\n",
    "df.rename(columns={\n",
    "    \"EdLevel\": \"Education Level\",\n",
    "    \"YearsCode\": \"Years of Experience\",\n",
    "    \"ConvertedComp\": \"Salary\"\n",
    "}, inplace=True)\n",
    "\n",
    "# View the result\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "392f3d63",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>Country</th>\n",
       "      <th>Education Level</th>\n",
       "      <th>Years of Experience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>United States</td>\n",
       "      <td>Bachelor’s degree (B.A., B.S., B.Eng., etc.)</td>\n",
       "      <td>17</td>\n",
       "      <td>116000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>United Kingdom</td>\n",
       "      <td>Master’s degree (M.A., M.S., M.Eng., MBA, etc.)</td>\n",
       "      <td>8</td>\n",
       "      <td>32315.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>United Kingdom</td>\n",
       "      <td>Bachelor’s degree (B.A., B.S., B.Eng., etc.)</td>\n",
       "      <td>10</td>\n",
       "      <td>40070.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Spain</td>\n",
       "      <td>Some college/university study without earning ...</td>\n",
       "      <td>7</td>\n",
       "      <td>14268.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>Netherlands</td>\n",
       "      <td>Secondary school (e.g. American high school, G...</td>\n",
       "      <td>35</td>\n",
       "      <td>38916.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Country                                    Education Level  \\\n",
       "7    United States       Bachelor’s degree (B.A., B.S., B.Eng., etc.)   \n",
       "9   United Kingdom    Master’s degree (M.A., M.S., M.Eng., MBA, etc.)   \n",
       "10  United Kingdom       Bachelor’s degree (B.A., B.S., B.Eng., etc.)   \n",
       "11           Spain  Some college/university study without earning ...   \n",
       "12     Netherlands  Secondary school (e.g. American high school, G...   \n",
       "\n",
       "   Years of Experience    Salary  \n",
       "7                   17  116000.0  \n",
       "9                    8   32315.0  \n",
       "10                  10   40070.0  \n",
       "11                   7   14268.0  \n",
       "12                  35   38916.0  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Preprocessing: Drop rows with missing values (if any)\n",
    "df.dropna(inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "25e65f76",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 34070 entries, 7 to 64154\n",
      "Data columns (total 4 columns):\n",
      " #   Column               Non-Null Count  Dtype  \n",
      "---  ------               --------------  -----  \n",
      " 0   Country              34070 non-null  object \n",
      " 1   Education Level      34070 non-null  object \n",
      " 2   Years of Experience  34070 non-null  object \n",
      " 3   Salary               34070 non-null  float64\n",
      "dtypes: float64(1), object(3)\n",
      "memory usage: 1.3+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a97e7fb4",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[df[\"Salary\"] <= 250000]\n",
    "df = df[df[\"Salary\"] >= 10000]\n",
    "df = df[df['Country'] != 'Other']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5a510078",
   "metadata": {},
   "outputs": [],
   "source": [
    "def shorten_categories(categories, cutoff):\n",
    "    categorical_map = {}\n",
    "    for i in range(len(categories)):\n",
    "        if categories.values[i] >= cutoff:\n",
    "            categorical_map[categories.index[i]] = categories.index[i]\n",
    "        else:\n",
    "            categorical_map[categories.index[i]] = 'Other'\n",
    "    return categorical_map"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d8bf3d2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Country\n",
       "Other                 8086\n",
       "United States         7266\n",
       "United Kingdom        2249\n",
       "Germany               2069\n",
       "India                 1267\n",
       "Canada                1216\n",
       "France                1043\n",
       "Brazil                 843\n",
       "Netherlands            771\n",
       "Poland                 749\n",
       "Australia              683\n",
       "Spain                  679\n",
       "Italy                  607\n",
       "Sweden                 552\n",
       "Russian Federation     529\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "country_map = shorten_categories(df.Country.value_counts(), 400)\n",
    "df['Country'] = df['Country'].map(country_map)\n",
    "df.Country.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cadef383",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean \"Years of Experience\" (some values are like 'Less than 1 year' or 'More than 50 years')\n",
    "def clean_experience(x):\n",
    "    if x == 'Less than 1 year':\n",
    "        return 0.5\n",
    "    if x ==  'More than 50 years':\n",
    "        return 50\n",
    "    return float(x)\n",
    "\n",
    "df['Years of Experience'] = df['Years of Experience'].apply(clean_experience)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5a9f3e1e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([17. ,  8. , 10. ,  7. , 35. ,  5. , 37. ,  9. , 30. ,  4. , 19. ,\n",
       "       20. , 25. , 16. , 36. ,  6. , 43. , 23. , 11. , 38. , 24. , 21. ,\n",
       "        3. , 40. , 15. , 27. , 12. , 46. , 13. , 14. , 33. , 22. , 18. ,\n",
       "       28. , 32. , 44. , 26. , 42. ,  2. , 34. , 31. , 29. ,  1. , 41. ,\n",
       "       50. , 47. , 39. ,  0.5, 45. , 48. , 49. ])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"Years of Experience\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a968077e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_education(x):\n",
    "    if 'Bachelor’s degree' in x:\n",
    "        return 'Bachelor’s degree'\n",
    "    if 'Master’s degree' in x:\n",
    "        return 'Master’s degree'\n",
    "    if 'Professional degree' in x or 'Other doctoral' in x:\n",
    "        return 'Post grad'\n",
    "    return 'Less than a Bachelors'\n",
    "\n",
    "df['Education Level'] = df['Education Level'].apply(clean_education)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5e8fe1b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Bachelor’s degree', 'Master’s degree', 'Less than a Bachelors',\n",
       "       'Post grad'], dtype=object)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"Education Level\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f15b5cb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reduce the Salary and Years of Experience column to float32 for memory efficiency\n",
    "df[\"Salary\"] = df[\"Salary\"].astype(\"float32\")\n",
    "df[\"Years of Experience\"] = df[\"Years of Experience\"].astype(\"float32\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8ef32c71",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([14, 13, 11,  7,  4,  2,  8,  6,  1,  3, 12,  5,  9,  0, 10])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Label Encoding for categorical column of \"Country\"\n",
    "le_country = LabelEncoder()\n",
    "df[\"Country\"] = le_country.fit_transform(df[\"Country\"])\n",
    "df[\"Country\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "29970a3f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 2, 1, 3])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Label Encoding for categorical column of \"Education Level\"\n",
    "le_education = LabelEncoder()\n",
    "df[\"Education Level\"] = le_education.fit_transform(df[\"Education Level\"])\n",
    "df[\"Education Level\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "dd1ab44f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define features and label\n",
    "X = df.drop(\"Salary\", axis=1)\n",
    "y = df[\"Salary\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b09837a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hillo\\AppData\\Local\\Temp\\ipykernel_6704\\1452971847.py:3: FutureWarning: \n",
      "\n",
      "The `ci` parameter is deprecated. Use `errorbar=None` for the same effect.\n",
      "\n",
      "  sns.barplot(data=df, x='Country', y='Salary', estimator='mean', ci=None, palette='plasma')\n",
      "C:\\Users\\hillo\\AppData\\Local\\Temp\\ipykernel_6704\\1452971847.py:3: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.barplot(data=df, x='Country', y='Salary', estimator='mean', ci=None, palette='plasma')\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArEAAAGGCAYAAABsTdmlAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjMsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvZiW1igAAAAlwSFlzAAAPYQAAD2EBqD+naQAATHpJREFUeJzt3Qm8jPX7//HLvpU9W9kq2SNrWqgIESllrVTSJlkqS8iSEoUUUamolOUbkmT5opXsyt4mZK1k37n/j/fn/7vnO3McnMM5Zm5ez8djnDMz99z358wZc97zua/7ulN4nucZAAAAECApoz0AAAAAILEIsQAAAAgcQiwAAAAChxALAACAwCHEAgAAIHAIsQAAAAgcQiwAAAAChxALAACAwCHEAgAAIHAIsQAQcF999ZWlSJHCfY0VDzzwgF100UXRHgaA8xghFkCyePPNN12wqly5crSHEnMOHz5sgwcPtmuuucYyZ85sWbNmtZIlS9ojjzxia9asifbwAufYsWP2/vvv20033WTZs2e3dOnSWaFChezBBx+0RYsWWSxYtWqV9ezZ0/74449oDwU4b6SO9gAAnJ9Gjx7tgsSCBQvs119/tSuvvDLaQ4oZDRs2tC+//NKaNm1qrVq1siNHjrjwOmXKFLvuuuusWLFi0R5iYBw4cMDuuusumzZtmlWtWtWee+45F2QVFseNG2ejRo2yDRs22GWXXRb1ENurVy8XtPX/AsDZI8QCSHLr1q2zuXPn2oQJE+zRRx91gbZHjx7ndAzHjx93M57p06e3WLJw4UIXVl988UUXuMINGTLEdu7cadG2b98+y5QpkwXBs88+6wLsoEGDrF27dhH36TWn24PG8zw7ePCgZciQIdpDAWIa5QQAkpxCa7Zs2axu3bp29913u+s+zTpqpky7euPavXu3C53PPPNM6LZDhw65MKKZXO0mzp8/v3Xs2NHdHk6lC08++aTblnbNa1mFG3n11VfdDGeOHDlcMChfvrz95z//iXdW76mnnrKcOXPaxRdfbPXr17dNmza5dWtXcDjd/tBDD1nu3LndtrTN995777TPzW+//ea+Xn/99SfclypVKjdG3/r16+2JJ56wokWLunHrvnvuuSdBu6S//fZbt2yBAgVCz1v79u3dzxhf7arGVadOHfdzN2/e3D3nadKksb/++uuEdavsQSUQClqn8/vvv1utWrVcKM6XL5/17t3bhTTRV81K3nHHHSc8TuvOkiWL+xB0Mn/++ae99dZbduutt54QYP3nU6+l8FnYpUuX2m233ebKOPRzV69e3X744YeIx+l3rd95XCNHjnS3hz//Gv/tt99u3333nVWqVMm9fi+//HL74IMPIh6n34XcfPPNbh3hNcz+OqZPn24VKlRwv2v9XNWqVbMyZcrE+7PrNaHnFbiQEWIBJDkFSe3iTZs2rdtl/ssvv7gZSFEwuvPOO23SpElupjScblM4bdKkSWg2VUFSIbRevXr2xhtvWIMGDdzsWuPGjU/Y7uzZs11Q032qOfV32/r1pwpQL730kqVOndqFii+++OKEQKdtKMz169fPhQkF8bi2bdtm1157rf33v/91wVnrV8hu2bKlvfbaa6d8bgoWLBh6jo4ePXrKZfWcaUZbz8frr79ujz32mM2aNcvtkt6/f/8pHzt+/Hi3zOOPP+5+JgUefb3//vtPWFbj0P25cuVyz7XKHe677z53+9ixYyOW1e9MHwC0zOlmuVWrWrt2bRf0+/fv7z48KBz7s/IKcvfee68rrdixY0fEYz///HP3oUb3n4wepzFqrAmxcuVKu/HGG+3HH390H4S6d+/u9hro+Zw/f76dKZXL6MOawvSAAQPcBzi9lrQ9UZmDPhyJZt8//PBDdylevHhoHWvXrnX/V7QOvZ7Kli3rfq6ffvrJVqxYccLr4ueffz7lcwNcEDwASEKLFi3SNJs3c+ZMd/348ePeZZdd5rVt2za0zPTp090yn3/+ecRj69Sp411++eWh6x9++KGXMmVK79tvv41Ybvjw4e7x33//feg2XdeyK1euPGFM+/fvj7h++PBhr1SpUt4tt9wSum3x4sVuHe3atYtY9oEHHnC39+jRI3Rby5Ytvbx583p///13xLJNmjTxsmTJcsL2wun5qFatmltn7ty5vaZNm3pDhw711q9ff9pxy7x589xjP/jgg9Btc+bMcbfp66ke27dvXy9FihQR22rRooV7bOfOnU9YvkqVKl7lypUjbpswYcIJ24qPv942bdpE/Ox169b10qZN6/3111/utrVr17rlhg0bFvH4+vXre4UKFXKPOZn27du7xy5dutRLiAYNGrht//bbb6HbNm/e7F188cVe1apVQ7fpdx3fn8f333/f3b5u3brQbQULFnS3ffPNN6Hb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TopxAQAAAEkXYv/66y978MEH7csvv4z3fmpiAQAAkNwS3Z1AJxdQP1adFStDhgyuJ+uoUaNc39TJkycnzygBAACAs5mJ1SlbP/vsM9cvVicOKFiwoN16662uj1ffvn3d2awAAACAmJqJ1dmxcuXK5b7XWbtUXiClS5e2JUuWJP0IAQAAgLMNsUWLFrW1a9e678uUKWNvvfWWbdq0yYYPH2558+ZN7OoAAACA5C8naNu2rTtTl/To0cNq165to0ePtrRp09rIkSMTPwIAAAAguUPsvffeG/q+fPnytn79eluzZo0VKFDAcubMmdjVAWfkoVxvRHX7721vE9XtAwBwoTvjPrG+jBkzWrly5ZJmNAAAAEBShdgOHTpYQg0cODDBywIAAFxI/pxePWrbvqzWLLvgQuzSpUsTtLIUKVKc7XgAAACApAmxc+bMSchiAAAAQGy22AIAAAACeWDXokWLbNy4cbZhwwY7fPhwxH0TJkxIqrEBAAAASTMTO2bMGLvuuuts9erVNnHiRDty5IitXLnSnY42S5YsiV0dAAAAkPwh9qWXXrJBgwbZ559/7k5wMHjwYNcntlGjRq5XLAAAABBzIfa3336zunXruu8VYvft2+e6ErRv397efvvt5BgjAAAAcHYhNlu2bLZnzx73/aWXXmorVqxw3+/cudP279+f2NUBAAAAyX9gV9WqVW3mzJlWunRpu+eee6xt27auHla3Va8evQa+AAAAuHAkOsQOGTLEDh486L7v2rWrpUmTxubOnWsNGza0bt26JccYAQAAgLMLsdmzZw99nzJlSuvcuXNiVwEAAACcmxB79OhRO3bsmKVLly5027Zt22z48OHu4K769evbDTfccHajAQAAAJIyxLZq1cp1I3jrrbfcdR3cVbFiRVdakDdvXtd267PPPrM6deokdJUAAABA8nYn+P77713dq++DDz5wM7O//PKL/fjjj9ahQwd75ZVXzmwUAAAAQHKE2E2bNlmRIkVC12fNmuVCrX+WrhYtWrgzdwEAAAAxU06QPn16O3DgQOj6Dz/8EDHzqvv37t2b9CMEkKT6Xvla1Lbd5dd2Uds2AOACnYktW7asffjhh+77b7/91h3Udcstt0ScyStfvnzJM0oAAADgTGZin3/+ebvtttts3LhxtmXLFnvggQfcAV2+iRMn2vXXX5/Q1QEAksh/m3aP6vZrfPJCVLcP4MKU4BBbrVo1W7x4sc2YMcPy5MnjztYVd6a2UqVKyTFGAAAA4MxPdlC8eHF3ic8jjzySmFUBAAAAyV8TCwAAAMQKQiwAAAAChxALAACAwCHEAgAA4MIIsTt37rQRI0ZYly5dbMeOHe62JUuWuLN6JTWt895777UcOXJYhgwZrHTp0rZo0aLQ/Z7nufZfavel+2vUqOFOhRtOY2zevLllzpzZsmbNai1btjzhxAw//fST3Xjjje6kDfnz57f+/fufMJbx48dbsWLF3DIax9SpU5P85wUAAEAyhFiFvauuusr69etnr776qgu0MmHCBBdqk9K///7res+mSZPGvvzyS1u1apUNGDDAsmXLFlpGYfP111+34cOH2/z58y1TpkxWq1YtO3jwYGgZBVidEnfmzJk2ZcoU++abbyK6Kezevdtq1qxpBQsWdG3EdCaynj172ttvvx1aZu7cuda0aVMXgJcuXWoNGjRwlxUrViTpzwwAAIBkCLEdOnRwJzrQbKdmJH116tRx4TApKShrVvT99993PWgLFy7swuYVV1wRmoV97bXXrFu3bnbHHXfY1VdfbR988IFt3rzZJk2a5JZZvXq1TZs2zc0cV65c2W644QZ74403bMyYMW45GT16tB0+fNjee+89K1mypDVp0sSeeuopGzhwYGgsgwcPttq1a9uzzz7r2oy98MILVq5cORsyZEiS/swAAABIhhC7cOFCe/TRR0+4/dJLL7WtW7daUpo8ebJVqFDBnVghV65cds0119g777wTun/dunVumyoh8GXJksWF1Xnz5rnr+qoSAq3Hp+VTpkzpZm79ZapWrWpp06YNLaPZ3LVr17rZYH+Z8O34y/jbAQAAQAyH2HTp0rnd73H9/PPPdskll1hS+v33323YsGFWpEgRmz59uj3++ONuhnTUqFHufj80586dO+Jxuu7fp68KwOFSp05t2bNnj1gmvnWEb+Nky5wquB86dMg9V+EXAAAARCHE1q9f33r37m1Hjhxx11OkSGEbNmywTp06WcOGDS0pHT9+3O2yf+mll9wsrOpYW7Vq5epfg6Bv375uZti/qDQCAAAAUQixOrBKR/ZrdvPAgQNWrVo1u/LKK+3iiy+2F1980ZKSOg6UKFEi4jbVoyo0S548edzXbdu2RSyj6/59+rp9+/aI+48ePeo6FoQvE986wrdxsmX8++OjA9127doVumzcuDGRzwAAAACSJMRqRlFH+X/++eeuK8CTTz7pWk19/fXXrjNAUlJnAtWlxi1bUBcB0YFeCpGzZs0K3a9d9qp1rVKliruur+qgoK4DvtmzZ7tZXtXO+svooDR/dln0MxYtWjTUCUHLhG/HX8bfzslKL9TWK/wCAACAs5f6TB+oo/x1SU7t27e36667zpUTNGrUyBYsWODaXvmtr1TK0K5dO+vTp4+rm1Wo7d69u+XLl8+1v/JnbtVVwC9DUFBV8FYHAi0nzZo1s169ern2WSqLUNssdSMYNGhQaCxt27Z1s86aia5bt67rbqB+teFtuAAAABCjIVazr/FRoFTLLZUW6Ej/VKlSnfXgKlasaBMnTnS75VWHq5Cqllrq++rr2LGj7du3z9XLasZVwVottcLbf6mFloJr9erVXVcC1e6G/xyaXZ4xY4a1bt3aypcvbzlz5nQnUAjvJasw/fHHH7t2Xs8995wLzWrjVapUqbP+OQEAQNLYNKZeVLd/aZPPo7r9C0miQ6xmJ//66y/bv39/aFe72lBlzJjRLrroIld/evnll9ucOXOS5ECm22+/3V1ORuFZAVeXk1EnAgXQU1GP2W+//faUy6jVly4AAAAIWE2sdu1rhlQnO/jnn3/cRXWqqi/VLngddKU6VZUCAAAAADExE6vd6Z9++mnorFmiEgKdgla76dXbVaeCTep2WwAAAMAZz8Ru2bLFtaiKS7f5jf91wNSePXsSu2oAAAAgeULszTff7E47u3Tp0tBt+l5n07rlllvc9eXLl7uDsAAAAICYCLHvvvuuO1BKR/GrD6ouFSpUcLfpPtEBXmpFBQAAAMRETawO2lKT/zVr1rgDukQnBdAlfLYWAAAAiLmTHRQrVsxdAAAAgECE2D///NMmT57s2mkdPnw44r6BAwcm1dgAAACApAmxs2bNsvr167sTGqikQGes+uOPP8zzPCtXrlxiVwcAAAAk/4FdOgXsM8884zoQ6NSu6hm7ceNGq1atGmezAgAAQGyG2NWrV9v999/vvk+dOrUdOHDAdSPQaV/79euXHGMEAAAAzi7EZsqUKVQHmzdvXvvtt99C9/3999+JXR0AAACQ/DWx1157rX333XdWvHhxq1Onjj399NOutGDChAnuPgAAACDmQqy6D+zdu9d936tXL/f92LFjrUiRInQmAAAAQOyF2GPHjrn2WldffXWotGD48OHJNTYAAADg7ENsqlSprGbNmu7grqxZsybmocAF49m8Q6K6/Ve2PBnV7QMAEJMHdqkv7O+//548owEAAACSI8T26dPH9YmdMmWKbdmyxXbv3h1xAQAAAGLuwC51JBCdtStFihSh23XGLl1X3SwAAAAQUyF2zpw5yTMSAAAAILlCrE4vCwAAAASqJla+/fZbu/fee+26666zTZs2uds+/PBDdxIEAAAAIOZmYj/99FO77777rHnz5rZkyRI7dOiQu33Xrl320ksv2dSpU5NjnACAAFrYulNUt19xaL+obh9AjHUn0AkO3nnnHUuTJk3o9uuvv96FWgAAACDmQuzatWutatWqJ9yeJUsW27lzZ1KNCwAAAEi6EJsnTx779ddfT7hd9bCXX355YlcHAAAAJH+IbdWqlbVt29bmz5/v+sJu3rzZRo8e7U6A8Pjjjyd+BAAAAEByH9jVuXNnO378uFWvXt3279/vSgvSpUvnQmybNm0SuzoAAAAg+UOsZl+7du1qzz77rCsr2Lt3r5UoUcIuuuiixG8dAAAAOBflBB999JGbgU2bNq0Lr5UqVSLAAgAAILZDbPv27S1XrlzWrFkz1xP22LFjyTMyAAAAIKlC7JYtW2zMmDGurKBRo0aWN29ea926tc2dOzexqwIAAADOTYhNnTq13X777a4jwfbt223QoEH2xx9/2M0332xXXHHFmY0CAAAASM4Du8JlzJjRatWqZf/++6+tX7/eVq9efTarAwAAAJJnJlZ0YJdmYuvUqWOXXnqpvfbaa3bnnXfaypUrz2R1AAAAQPLOxDZp0sSmTJniZmFVE9u9e3erUqVKYlcDAAAAnLsQmypVKhs3bpwrI9D34VasWGGlSpU689EAAAAAyRFiVUYQbs+ePfbJJ5/YiBEjbPHixbTcAgDgPLdheJOobbvAY2Oitm2cBzWx8s0331iLFi1ci61XX33VbrnlFvvhhx8sOb388suutVe7du1Ctx08eNC1+MqRI4c76ULDhg1t27ZtEY/bsGGD1a1b15VAqMetzjZ29OjRiGW++uorK1eunDuF7pVXXmkjR448YftDhw61QoUKWfr06a1y5cq2YMGCZPxpAQAAkCQhduvWrS5IFilSxO655x7LnDmzHTp0yCZNmuRur1ixoiWXhQsX2ltvvWVXX331CSdf+Pzzz238+PH29ddf2+bNm+2uu+4K3a+ZYQXYw4cPu162o0aNcgH1+eefDy2zbt06t4zahC1btsyF5IcfftimT58eWmbs2LHWoUMH69Gjhy1ZssTKlCnjSirUZgwAAAAxGmLr1atnRYsWtZ9++sl1I1BYfOONN+xc2Lt3rzVv3tzeeecdy5YtW+j2Xbt22bvvvmsDBw50M8Hly5e3999/34VVf1Z4xowZtmrVKne63LJly9ptt91mL7zwgptVVbCV4cOHW+HChW3AgAFWvHhxe/LJJ+3uu+92PXB92karVq3swQcfdKfb1WM0s/vee++dk+cAAAAAZxBiv/zyS2vZsqX16tXLzVrGPagrOalcQNusUaNGxO2qwT1y5EjE7cWKFbMCBQrYvHnz3HV9LV26tOXOnTu0jGZQd+/eHWoJpmXirlvL+OtQ2NW2wpdJmTKlu+4vAwAAgBgMsd999507iEuznaoHHTJkiP3999/JOzozd4pb7b7v27dvvOUNadOmtaxZs0bcrsCq+/xlwgOsf79/36mWUdA9cOCA+zlVlhDfMv464qNSC60j/AIAAIBzGGKvvfZatzt/y5Yt9uijj7pwmS9fPjt+/LjNnDnTBdyktnHjRmvbtq3riKCDqYJGwTtLliyhS/78+aM9JAAAgAuzxVamTJnsoYcecpe1a9e6mlQd1NW5c2e79dZbbfLkyUk2OO3C14FT6hrg04yoOiNoJlgHXmlX/86dOyNmY9WdIE+ePO57fY3bRcDvXhC+TNyOBrquA9cyZMjgSid0iW8Zfx3x6dKlizsYzKeZWIIsEEz/qdUnatu+e3q3qG0bAM6bEBtOB3r179/fzTiqQ0BSH+RUvXp1W758ecRtOrBKda+dOnVygTBNmjQ2a9Ys11pLFKzVUss/i5i+vvjiiy4Mq72WaOZYAVUHaPnLTJ06NWI7WsZfh0oWVEah7TRo0MDdphloXddBYCejdl26AABi06quT0Vt2yVefD1q2wbsQg+xPs1SKtz5AS+pXHzxxSecAUwzweoJ69+ug80025k9e3YXTNu0aePCp8ofpGbNmi6s3nfffS5wq4a1W7du7mAxP2A+9thjbma3Y8eOboZ59uzZ7qxkX3zxRWi72ob64laoUMEqVarkOjTs27fPhWoAAAAEMMRGk9pgqVOAZmJ1IJW6Crz55psRAXvKlCn2+OOPu3CrEKww2rt379Ayaq+lwKqes4MHD7bLLrvMnYFM6/I1btzY/vrrL9dfVkFY7bqmTZt2wsFeAAAASH6BC7E6s1Y4HfClnq+6nEzBggVPKBeI66abbrKlS5eechmVDpyqfAAAAAAxftpZAAAAIFoIsQAAAAgcQiwAAAAChxALAACAwCHEAgAAIHAIsQAAAAgcQiwAAAAChxALAACAwCHEAgAAIHACd8YuAAAuBL/1axm1bV/R6d2obRtIKGZiAQAAEDiEWAAAAAQOIRYAAACBQ4gFAABA4BBiAQAAEDiEWAAAAAQOIRYAAACBQ4gFAABA4BBiAQAAEDicsQtAzHi74itR2/YjC5+N2rYBAInHTCwAAAAChxALAACAwCHEAgAAIHCoicVJ3ZF7YNS2/dm2DlHbNgAAiH3MxAIAACBwCLEAAAAIHEIsAAAAAocQCwAAgMAhxAIAACBwCLEAAAAIHEIsAAAAAocQCwAAgMAhxAIAACBwCLEAAAAIHEIsAAAAAocQCwAAgMAhxAIAACBwCLEAAAAInJgOsX379rWKFSvaxRdfbLly5bIGDRrY2rVrI5Y5ePCgtW7d2nLkyGEXXXSRNWzY0LZt2xaxzIYNG6xu3bqWMWNGt55nn33Wjh49GrHMV199ZeXKlbN06dLZlVdeaSNHjjxhPEOHDrVChQpZ+vTprXLlyrZgwYJk+skBAAAQ2BD79ddfu4D6ww8/2MyZM+3IkSNWs2ZN27dvX2iZ9u3b2+eff27jx493y2/evNnuuuuu0P3Hjh1zAfbw4cM2d+5cGzVqlAuozz//fGiZdevWuWVuvvlmW7ZsmbVr184efvhhmz59emiZsWPHWocOHaxHjx62ZMkSK1OmjNWqVcu2b99+Dp8RAAAASOpYfhqmTZsWcV3hUzOpixcvtqpVq9quXbvs3XfftY8//thuueUWt8z7779vxYsXd8H32muvtRkzZtiqVavsv//9r+XOndvKli1rL7zwgnXq1Ml69uxpadOmteHDh1vhwoVtwIABbh16/HfffWeDBg1yQVUGDhxorVq1sgcffNBd12O++OILe++996xz587n/LkBAAC4kMX0TGxcCq2SPXt291VhVrOzNWrUCC1TrFgxK1CggM2bN89d19fSpUu7AOtTMN29e7etXLkytEz4Ovxl/HVoFlfbCl8mZcqU7rq/THwOHTrkthN+AQAAwAUUYo8fP+52819//fVWqlQpd9vWrVvdTGrWrFkjllVg1X3+MuEB1r/fv+9Uyyh0HjhwwP7++29XlhDfMv46TlbTmyVLltAlf/78Z/UcAAAAIGAhVrWxK1assDFjxlhQdOnSxc0e+5eNGzdGe0gAAADnhZiuifU9+eSTNmXKFPvmm2/ssssuC92eJ08et6t/586dEbOx6k6g+/xl4nYR8LsXhC8Tt6OBrmfOnNkyZMhgqVKlcpf4lvHXER91OtAFAAAAF9BMrOd5LsBOnDjRZs+e7Q6+Cle+fHlLkyaNzZo1K3SbWnCppVaVKlXcdX1dvnx5RBcBdTpQQC1RokRomfB1+Mv461DJgrYVvozKG3TdXwYAAADnTupYLyFQ54HPPvvM9Yr1609VX6oZUn1t2bKla32lg70UTNu0aeOCpToTiFpyKazed9991r9/f7eObt26uXX7s6SPPfaYDRkyxDp27GgPPfSQC8zjxo1z3Qd82kaLFi2sQoUKVqlSJXvttddcqy+/WwEAAADOnZgOscOGDXNfb7rppojb1UbrgQcecN+rDZY6BegkB+oGoK4Cb775ZmhZlQGoFOHxxx934TZTpkwujPbu3Tu0jGZ4FVjVc3bw4MGuZGHEiBGh9lrSuHFj++uvv1x/WQVhtepSC7C4B3sBAADgAg+xKic4HZ09S2fS0uVkChYsaFOnTj3lehSUly5desplVNqgCwAAAKIrpmtiAQAAgPgQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hFgAAAAEDiEWAAAAgUOIBQAAQOAQYgEAABA4hNhEGjp0qBUqVMjSp09vlStXtgULFkR7SAAAABccQmwijB071jp06GA9evSwJUuWWJkyZaxWrVq2ffv2aA8NAADggkKITYSBAwdaq1at7MEHH7QSJUrY8OHDLWPGjPbee+9Fe2gAAAAXlNTRHkBQHD582BYvXmxdunQJ3ZYyZUqrUaOGzZs3L97HHDp0yF18u3btcl93794duu3o8YMWLeHjiM+RGB7b4eMHLJpONb5DMTw2ORjDv9cDx2J3bPuPxu7Y9h353/tMrI1v7+HYHZvsPXTYYnVsew7G8NgOHLGYHdv+6I0tQePbd9Ridmx7Y2Ns/vee553yMSm80y0BZ/PmzXbppZfa3LlzrUqVKqHbO3bsaF9//bXNnz//hMf07NnTevXqdY5HCgAAEHwbN260yy677KT3MxObjDRrqxpa3/Hjx23Hjh2WI0cOS5EixVmvX59U8ufP737JmTNntljC2M4MYzs/x8fYzgxjOz/Hx9jOzIU0Ns/zbM+ePZYvX75TLkeITaCcOXNaqlSpbNu2bRG363qePHnifUy6dOncJVzWrFmTfGx6wcTaC9rH2M4MYzs/x8fYzgxjOz/Hx9jOzIUytixZspx2GQ7sSqC0adNa+fLlbdasWREzq7oeXl4AAACA5MdMbCKoNKBFixZWoUIFq1Spkr322mu2b98+160AAAAA5w4hNhEaN25sf/31lz3//PO2detWK1u2rE2bNs1y584dlfGoVEE9a+OWLMQCxnZmGNv5OT7GdmYY2/k5PsZ2ZhjbiehOAAAAgMChJhYAAACBQ4gFAABA4BBiAQAAEDiEWAAAAAQOIRYAAACBQ4iNMTqBwrFjx6I9jECj4UbibNmyxVatWmWxyP+/EIu/0/3799vhw4ctVv3555+2dOnSaA8jkO/BugCIfYTYGKIgcf/991utWrXs8ccft7lz51osieVwrZNO6DzLOn9zihQpLJbs2LHD1qxZY7/88kvMhZ5NmzZZ6dKlrVu3brZo0SKLJcuWLbMGDRq4sBhrv9MVK1ZYo0aN7IcffrBDhw5ZrFm5cqVdd9119tFHH7nrsRTKFK7HjRtnEyZMsOXLl1usvQc/8MADVqNGDXvkkUdszJgxFuti8QMeEvf7i9W/rTt27HC98WMZITZGrF271v3R0Yu5YsWKNm/ePGvbtq29/vrrFgt+/vlnd4YyzdrFGv3hueuuu6xatWpWvHhxGz16dMy8uSvs6A+iAo/CYv/+/WPqDUvBeteuXe7yxhtv2JIlS0L3RfP5+/HHH93/h5IlS1rGjBljYkzhAfHGG2+0yy67zAoXLhxzjcf13OmMgqlTp7aPP/7Ytm/fbilTxsZbvULrDTfcYK+88oo98cQT1rVrV/vtt98sFuiDpsamU4zffvvttmHDBuvevbu1adPGYuU9uFOnTu4MkYMHD3b/d0Uf8KL9/0KvsZ07d1osWrdunQ0aNMiefvppGzt2rMUS/U7bt29vd9xxh/Xu3dv++ecfixW///67yyL6u7B582aLWTrZAaLr+PHj3nPPPec1atQodNvu3bu9Pn36eGXLlvX69esX1fH98ssvXvbs2b0UKVJ4Xbp08f766y8vVqxcudLLkSOH1759e2/06NFehw4dvDRp0nhLly6NmbE988wz7vtXX33VPYcbNmzwYsU///zj1a9f33vrrbe8cuXKec2bN/dWrFjh7jt27FhUxvTjjz96mTJl8p599tmI2w8dOuRF2969e72aNWt6jz/+eOi21atXu9fb+vXrvWhbtmyZlyFDBvd+ov+nJUuWdO8jeo/RJZr++OMP79JLL/U6d+7snsepU6d6efLk8ebPn+9F28GDB91r/6mnngrdduDAAe+aa65x/2ebNm0a1fHp/SNLlixe7dq1vYYNG7rva9So4b3zzjuhZaL1+121apWXNm1a7+677/Z27drlxZKffvrJu+yyy7zq1at71113nZcyZUqvf//+XqyMLVeuXO55e/TRR91z2LNnTy9WDB8+3L329X/gxRdf9LZs2RK6LxbeT3yE2BjxwAMPeFWrVo24TUFWwadChQreRx99FJVx6Y/NQw895MY3dOhQ96JWuIiFIKsApkAR/odHbrrpJq9Nmzbu+2j9R9Pzo99n27ZtQ7dpLPojNHfuXBd6oh1mjx496m3fvt276qqrvD///NObMGGCV7FiRa9Vq1buDV9/LM81vVEq2NSqVSs0xnbt2nl169b1ihUr5g0aNMiFxmiGnRtuuMFbsmSJG5vGqefs4osv9q699lpvxIgRURubwn+6dOlcgPU/hOgPpMbni+YfHn1Q0v/N8DHUqVPH3T5q1Chv9uzZXjQp6PghQgFWOnbs6P4f6APeK6+8EpVx6cPbvffe6/5fhk8sNG7c2L3mBg8e7EXL1q1b3XvFLbfc4uXMmdO75557YibI6kPTlVde6X6H/gfyd99918udO7f3888/R3Vsv//+u1eoUCE3KeTTa++JJ57wDh8+HLFstP7P/vjjj16LFi3ch+B8+fJ5L7zwgvfvv/96sSY29jFdwPzdQOXKlXO7mVVW4Lv44ovtoYcesmuuucbefPNNVxt4rmk3ZPny5a127dpu959qxF599VW3W/zvv/+2aDpy5IjbhXX33XdH1P1pF69qeSRatZTarp6z1q1bh27r06ePTZ8+3T2P9erVs1atWtl3331n0aLf7SWXXOJ2Gans4c4777SePXvaxIkT3W5f7VKNhipVqrjdap999pkbg8ZSrFgxq169uiuv0etPu3qjQa83/R/Va//ZZ591t40YMcLVeKrEQLXF//nPf6IyNtXmduzY0V588UX3f0G/X73mtMty2LBhbplo1hbrvU6/N9U6i8b55Zdf2vjx423IkCHWpEkTGzlyZFTG5R+kp9KGo0ePWvr06V29uHY/161b10qUKGFTp061aFB5w7Zt20K/O433yiuvdO/B+n+h19vnn38elbHpwMFChQpZv3797IsvvrBZs2bZww8/7I5NiCa9/vW3Ss/Tc889Fyqn0XtdmjRpolojrr/zn376qd12223WuXPnEw7EvP76690xMf7vNFr/Zz3Pc8fl6Pl79NFH7e2337ZRo0a50j2VAcWMaKdo/H+//vqr+ySrWc89e/ZEfALTjJ1mQL/88suozcaGGzNmjBuPdpP//fff7jZ90tWny3Mt/BO1/wm2W7du3n333RexnP+cnkuaSfd98skn7jkbO3asm0H++uuv3QxZLOw+uv/++90uXmnZsqWXLVs2r0SJEu61GI1dvZs3b3Zj0m7xW2+9NfQaE5WMZM2a1e2Kjgb9n2zSpIn35JNPerfffrs3bdq00H0bN250M2aPPfaYm6WN9u42bX/nzp1egwYNXKlStMek9wfN2ml2TLOb+v8wadIkN6Zt27a5PSqaqdXvOxrj/O6779zuZu1B0fuHSloefvhhd9/y5cvdbPuaNWvO6dj0O9P72oMPPuhm1bUnQNv3ZxZ/++03r0qVKm5WNhq0J2fOnDmh6/PmzXOlZ5qR1WvPF43fp95j/fc1n543zYCGjzka9F6h58qnWc5UqVJ5Xbt29V5//XX3t0Gz2+G78KOhZs2a3rp169z3KsPQ/wmVskyfPt2LFYTYGKLdadod2Lp164jd9XohlylTxu2GjqbwP4J+KFNpwaZNm1xN6l133eXt27cvKmMLr9/UG4G/O1peeukl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",
      "text/plain": [
       "<Figure size 700x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Average Salary by Country\n",
    "plt.figure(figsize=(7,4))\n",
    "sns.barplot(data=df, x='Country', y='Salary', estimator='mean', ci=None, palette='plasma')\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.title('Average Salary by Country')\n",
    "plt.ylabel('Average Salary')\n",
    "plt.xlabel('Country')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e1942fa1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArIAAAGGCAYAAACHemKmAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjMsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvZiW1igAAAAlwSFlzAAAPYQAAD2EBqD+naQAAUvtJREFUeJzt3Qm8VPMf//FP+6qFtGklLSi0SLRRP1GW5EeI+pH8ULaohF8U6aeUNUt2P0X1s/ySlKRFRKlQSUibpYUWSqvm/3h///8z/zPT3Hvn3jv3zj0zr+fjMc0953zvzJmZc5vP+Z7P9/MtFAqFQgYAAAAETOFk7wAAAACQEwSyAAAACCQCWQAAAAQSgSwAAAACiUAWAAAAgUQgCwAAgEAikAUAAEAgEcgCAAAgkAhkAQAAEEgEskAaaN++vbsF2dq1a61QoUL20ksv5flz6Tn0XHpOT506dezcc8+1/DBnzhz3/LrPb/n5OnO7n//4xz+sINNneO+99yZ7N9L27xjpgUAWKICWLVtmf//736127dpWsmRJO+qoo+xvf/ubPf7445Yq9GXm3YoWLWqHH364NWvWzG6++Wb7+uuvE/Y8Tz75ZIH90izI+5YfwUxGt3//+98WFNOmTStwwar2R+/jr7/+muxdAfJc0bx/CgDZ8cknn9gZZ5xhtWrVsj59+ljVqlVtw4YN9umnn9qjjz5qN954o6UKBec9e/a0UChkO3bssC+//NJefvllF+A9+OCD1r9//3BbBfW7d++2YsWKZes59FiVKlXKVu/dlVdeaZdeeqmVKFHC8lJG+9a2bVv3WosXL26p7LLLLrPOnTsfsv7kk0+2IAWyY8eOjRnM6jPUSRqAvMNfGFDADB8+3MqXL2+LFi2yChUqRGzbvHmzJduBAwfs4MGDCQmy6tevb1dccUXEOvXGnXfeeXbbbbdZw4YNw4GOepjUO52Xdu3aZWXKlLEiRYq4W7IULlw4z19rQdC0adNDPv9Ukg6fIZBspBYABczq1avt+OOPPySIlcqVK0csv/jii3bmmWe69eo9PO644+ypp57K8jn27dtnQ4YMcZfyFTQreGvTpo3Nnj075iXghx56yB555BE75phj3PMsXLjQ/Y7SAKL9+OOPLggcMWJEjl7/EUccYa+//rrryVJQH70v/kvxGzdutKuuuspq1Kjh9qtatWp2wQUXhHNblUe5YsUKmzt3bviytZcr7OXBatsNN9zg3kM9TkY5sp7333/fTjrpJBek6P1+8803Y17WjRb9mJntW0Y5spMnT3afWalSpVxProLAn376KaKNenfLli3r1nft2tX9fOSRR9rtt99uf/31V9yfQ2av84cffnD79/DDD8e8oqBtr732miWCeuvvv/9+99mULl3aXa3Q+xYt3vfd895771m7du3ssMMOs3LlylmLFi1swoQJ4e0fffSRXXzxxe7KiI6tmjVr2q233up6Wf3vtXpjxZ8akVmO7NKlS+2cc85xz6nPpkOHDu5qS6x9/vjjj91VCX1++nu78MILbcuWLZYo33zzjUthUlqPPufmzZvblClTwts///xztx+6ShJtxowZbtvUqVPD63TMXX311ValShX3nun/sRdeeCFh+wvEQo8sUMDoEvqCBQts+fLldsIJJ2TaVkGrvizOP/98F/i98847LihTj2nfvn0z/L3ff//dnnvuOXdpV+kLf/zxhz3//PPWqVMnF6QqgIkOmPfs2WPXXnut+4LSl7u+VCdOnGhjxoyJ6L1UAKPgo0ePHjl+D/T4CjIUWGtf9aUfy0UXXeSCGqVbKDBUj/XMmTNt/fr1blnBt7YpYLjrrrvc7+hL1k/vlwIFBfbqkc3Md999Z927d7frrrvOevXq5d4XBTvTp093aRLZEc++RQc3CtoVcOkkYdOmTS7VRMGOgiP/iY8CVn2WLVu2dCchH3zwgY0ePdqdiFx//fVZ7ltWr/Poo4+2008/3caPH++COz+tU3CoE4qs/PnnnzHzOPVavEvy+lwUyKpnXrclS5bYWWed5U7GckrvpQIu/e0MHjzYPZ/eQ72+yy+/PHzSoP3T+6WTK/1dKEddJ2raJv/85z/t559/dsfcf/7znyyfV8eqThh1PA8cONClyTzzzDPuBEYnNPq8/HR8VKxY0e655x4XiOuY6devn/u7yy3tiz5D5d/fcccdLlCeNGmSO/l544033N+3Alt91lqv48BP+6B903EmOh5PPfVUF9xqH/U3pZOF3r17u7/hW265Jdf7DMQUAlCgvP/++6EiRYq4W6tWrUIDBw4MzZgxI7Rv375D2v7555+HrOvUqVPo6KOPjljXrl07d/McOHAgtHfv3og227ZtC1WpUiV09dVXh9etWbMmpP8mypUrF9q8eXNEe+2T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VMJNg88069Nrr71GEAsgKeiRBQAAQCDRIwsAAIBAIpAFAABAIDEhQj7SfNSaMUezyiRy2kwAAIBUoazXP/74w6pXr37I5BvRCGTzkYLY/JpLGwAAIMg2bNhgNWrUyLQNgWw+Uk+s98Hk15zaAAAAQaIpy9Xx58VNmSGQzUdeOoGCWAJZAACAjMWThslgLwAAAAQSgSwAAAACiUAWAAAAgUQgCwAAgEAikAUAAEAgEcgCAAAgkAhkUWBMmDDB2rdvH75pGciJH374wc4880x3HOley0BOfPDBBxH/L2kZyImvv/464ljSMgIeyI4YMcJatGjhCt5WrlzZunbtaqtWrYpoow9bdcT8t+uuuy6izfr1661Lly5WunRp9zgDBgywAwcORLSZM2eONW3a1EqUKGH16tWzl1566ZD9GTt2rNWpU8dKlixpLVu2tIULF0Zs37Nnj/Xt29eOOOIIK1u2rF100UW2adOmhL4n6Uqf87hx4yLWaVnrgezQMXP11Ve7KaFF91rmWEJ26Zi5//77I9ZpmWMJ2aVj5oYbbohYp2WOpYAHsnPnznWB4aeffmozZ860/fv321lnnWW7du2KaNenTx/75ZdfwreRI0eGt/31118uiN23b5998skn9vLLL7sgdciQIeE2a9ascW3OOOMM++KLL+yWW26xa665xmbMmBFuM3HiROvfv7/dc889tmTJEjvxxBOtU6dOtnnz5nCbW2+91d555x2bPHmy23dNOdutW7c8f59SXVZ/yPyhI17+Y6VYsWIugNV9rO1AZqKPlfr162e6HchI9LFywQUXZLod2RQqQDZv3hzSLs2dOze8rl27dqGbb745w9+ZNm1aqHDhwqGNGzeG1z311FOhcuXKhfbu3euWBw4cGDr++OMjfq979+6hTp06hZdPOeWUUN++fcPLf/31V6h69eqhESNGuOXt27eHihUrFpo8eXK4zcqVK93+LliwIK7Xt2PHDtde9/i/xo8f7z7jrG5qB2Rm9erV4ePlp59+itimZW+b2gGZmTlzZvh4+eijjyK2adnbpnZAZlasWJHh/z3+/7PUDjmLlwpUjuyOHTvc/eGHHx6xfvz48VapUiU74YQTbPDgwfbnn3+Gty1YsMAaN25sVapUCa9TT6rm6V2xYkW4TceOHSMeU220XtSbu3jx4og2hQsXdsteG21Xj7G/TcOG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KMRq43+MrPYlVg6wZn7xbuohApC3KlWqdEiJLS1TKglAMp1yyimu401XinSvZaRQ1QLlyi5fvtz1oKSKwYMHu7xbjwJnglkgb+nEVJeBjzrqKHe1RSegP/30k7ssDADJooHp/o6vatWqJXV/UkWB6JHt16+fTZ061WbPnm01atQIr69atar7Itq+fXtEe1UK0DavTXTlAG85qzYavVyqVCnXg1OkSJGYbfyPkdW+RFOFBD2H/wYgb9x///3hn7/88kt744033OBM3Ws5VjsglgsvvDCh7ZC+nnjiifDPv/322yEn3bHaIUCBbCgUckHsW2+9ZR9++KEbkBWdFK2C5t786aIcJZ3VtGrVyi3rftmyZRHVBVQBQUHjcccdF27jfwyvjfcY6rHRc/nbKNVBy16bePYF2adSZolsh/Tlnxknuiajf5kZdJAVDTZOZDukLy9VUjTOxs+/7G+HAKUWKJ1AVQD+97//uVqyXpe78knVU6r73r17u8vzym9TcHrjjTe6wNGrEqByXQpYr7zyShs5cqR7DA3m0GOrR9Qrt6OznYEDB9rVV1/tguZJkya5qgQePYeKpmv+Y+WtPPLII64MmFfFIJ59QfZ580wnqh3Sl793IxHtkL6yM7OXRqADGdH4G9W4//PPPzNso5NrtdOVYQSsR/app55ylQrat2/vckW8m2q6elQi69xzz3WTD+jsV5fx33zzzfB2ffBKS9C9gsorrrjCevbsacOGDQu3UU+vglb1wqo+rMpwadSgV3pLunfvbg899JCrP6sSXqo3q/Jg/gFgWe0Lsi86VSO37ZC+tmzZktB2SF/PPvtsQtshfakuvYJYddY9+uijEdu0rPXqNFM75EyhkK7vI19osJd6dhW8ByFfVjOZKXUiL1177bVxtx03bpzlNZVT08xuCJ7bbrstXNJGaUAXX3yxde7c2aZNm+YmMfEu4ylNyKspDcSizpV4adZIICPDhw93nWg33XRTzAmUlMP/+OOP29/+9je76667krKPQY+XCkzVAhQ8CmKzE2jmtfzYFwXL9evXz/PnQd7VoRaVtlGVAk2Yov8Etez1ePjbAVnR8aMTI9UFV7qaToioCY54efn5utqs9IGvvvrKtm7d6lIUmzRpEh4sHp3Xj/gRyCLT3sm87gXVpblFixZl2a5FixZu5rX8eM0IpgMHDoR/9l+mi86J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FX3ttddcWsCVV15pI0eOjMgnLVOmTHiKWAAAAKDABLJff/21Pf74427WLf/Aq+g82mSV6QIAAEDqK5yTlAKlDZx66qkZBrHeVLHt2rXL7f4BAAAAiQlkixUrZm+88UZ2fw0AAABI/mAvTQKgOqsAAABAoHJkjz32WFdPVTNpqdSWBnf53XTTTYnaPwAAACBxgezzzz9vFSpUsMWLF7tb9MQBBLIAAAAokIGspnsFAAAAAj2zFwAAABCYHln58ccfbcqUKbZ+/Xrbt29fxLYxY8YkYt8AAACAxAays2bNsvPPP9+OPvpo++abb9ysXmvXrnUzfTVt2jQnDwkAAADkfWrB4MGD7fbbb7dly5ZZyZIlXV3ZDRs2uAkQLr744pw8JAAAAJD3gezKlSutZ8+e4Rm8du/ebWXLlnUluR588MGcPCQAAACQ94Gs6sZ6ebHVqlWz1atXh7f9+uuvOXlIAAAAIO9zZE899VSbP3++NWrUyDp37my33XabSzN488033TYAAACgQAayqkqwc+dO9/PQoUPdzxMnTnQzflGxAAAAAAU2kFW1An+awdNPP53IfQIAAACyxIQIAAAASO0e2YoVK1qhQoXiart169bc7BMAAACQuED2kUceibcpAAAAUHAC2V69euXtngAAAAB5PdjLb8+ePeGasp5y5crl9mEBAACAxA/22rVrl/Xr188qV67sqhYof9Z/AwAAAApkIDtw4ED78MMP7amnnrISJUrYc8895+rJVq9e3V555ZXE7yUAAACQiNSCd955xwWs7du3t6uuusratGlj9erVs9q1a9v48eOtR48eOXlYAAAAIG97ZFVey5sUQfmwXrmt1q1b27x583LykAAAAEDeB7IKYtesWeN+btiwoU2aNCncU1uhQoWcPCQAAACQ94Gs0gm+/PJL9/Mdd9xhY8eOtZIlS9qtt95qAwYMyMlDAgAAAHmfI6uA1dOxY0f75ptvbPHixS5PtkmTJjl5SAAAACDvemQXLFhgU6dOjVjnDfq67rrr7IknnrC9e/dmbw8AAACAvA5khw0bZitWrAgvL1u2zHr37u16ZQcPHuxyZEeMGJGT/QAAAADyLpD94osvrEOHDuHl119/3Vq2bGnPPvusSzd47LHHwgO/AAAAgAITyG7bts2qVKkSXp47d66dc8454eUWLVrYhg0bEruHAAAAQG4DWQWxXtmtffv22ZIlS+zUU08Nb//jjz+sWLFi2XlIAAAAIO8D2c6dO7tyWx999JHLiS1durSb1cvz1Vdf2THHHJOzPQEAAADyqvzWfffdZ926dbN27dpZ2bJl7eWXX7bixYuHt7/wwgt21llnZechAQAAgLwPZCtVquSmoN2xY4cLZIsUKRKxffLkyW49AAAAUCAnRChfvnzM9Ycffnhu9wcAAADIuylqAQAAgGQjkAUAAEAgEcgCAAAgkAhkAQAAEEgEsgAAAAgkAlkAAAAEUlIDWdWkPe+886x69epWqFAhe/vttyO2h0IhGzJkiFWrVs1KlSplHTt2tO+++y6izdatW61Hjx5Wrlw5q1ChgvXu3dt27twZ0UYzjmkGspIlS1rNmjVt5MiRh+yLauA2bNjQtWncuLFNmzYt2/sCAACANAlkd+3aZSeeeKKNHTs25nYFnI899pg9/fTT9tlnn1mZMmWsU6dOtmfPnnAbBbErVqywmTNn2tSpU11wfO2114a3//777262sdq1a9vixYtt1KhRdu+999q4cePCbT755BO77LLLXBC8dOlS69q1q7stX748W/sCAACAfBQqILQrb731Vnj54MGDoapVq4ZGjRoVXrd9+/ZQiRIlQq+99ppb/vrrr93vLVq0KNzmvffeCxUqVCj0008/ueUnn3wyVLFixdDevXvDbQYNGhRq0KBBePmSSy4JdenSJWJ/WrZsGfrnP/8Z977EY8eOHW5/dY/YVq1aFWrXrp27B3KDYwmJwrGEROFYSny8VGBzZNesWWMbN250l/D9M4q1bNnSFixY4JZ1r3SC5s2bh9uofeHChV2vqdembdu2Vrx48XAb9aSuWrXKtm3bFm7jfx6vjfc88ewLAAAAAjBFbX5Q4ChVqlSJWK9lb5vuK1euHLG9aNGibqpcf5u6dese8hjetooVK7r7rJ4nq32JZe/eve7mT3MAAABAYhTYHtlUMGLECNdz69000AwAAAApHshWrVrV3W/atClivZa9bbrfvHlzxPYDBw64Sgb+NrEew/8cGbXxb89qX2IZPHiw7dixI3zbsGFDtt4DAAAABDC1QOkAChJnzZplJ510UvjSvHJfr7/+erfcqlUr2759u6tG0KxZM7fuww8/tIMHD7r8Va/NXXfdZfv377dixYq5dapw0KBBA5dW4LXR89xyyy3h51cb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OTPi9+fMmeN6LLdv3x5ep55TrVNPqufjjz+29u3bW+nSpd3jd+rUybZt2+a2TZ8+3Vq3bu0unx9xxBF27rnn2urVq8O/W7duXXevud71uHqcWKkFe/futZtuuskqV67s9lePuWjRokP2ddasWda8eXO3L6eddpqtWrUqV+/hc889Z40aNXLP2bBhQ3vyySfD2/T4gwYNimivues15ee8efPC+3377bfbUUcd5aaVbdmypdtXAOmHQBYAEuTgwYPWrVs3K168uH322Wf29NNPHxKUxUOBbYcOHVwgvGDBAps/f76dd9559tdff7ntu3btsv79+9vnn3/ugszChQvbhRde6J5fFi5c6O4/+OAD++WXX+zNN9+M+TwDBw60N954w15++WVbsmSJ1atXzwXMW7dujWh311132ejRo93zFS1a1K6++mrLqfHjx9uQIUNs+PDhtnLlSnvggQfsX//6l9sH6dGjh73++uvmnz194sSJVr16dWvTpo1b7tevn3tf1O6rr76yiy++2PWU6yQCQJoJAQDi0qtXr1CRIkVCZcqUibgNHz7cbZ8xY0aoaNGioZ9++in8O++9954istBbb73llmfPnu2Wt23bFm6zdOlSt27NmjVu+bLLLgudfvrpce/Xli1b3O8vW7bMLetxtKzHjd7/Cy64wP28c+fOULFixULjx48Pb9+3b1+oevXqoZEjR0bs6wcffBBu8+6777p1u3fvznB//K832jHHHBOaMGFCxLr77rsv1KpVK/fz5s2b3Xs4b9688HZtGzRokPt53bp17jPwv8fSoUOH0ODBg93PL774Yqh8+fJZvGsAUgE5sgCQDWeccYY99dRTEesOP/xwd68expo1a7reQ0+rVq1y1COrXsaMqOdRvZrq9f3111/DPbHr16+3E044Ia7nUCrC/v377fTTTw+v0+X7U045xb0OvyZNmoR/rlatmrvfvHmz1apVK1uvSz3Jet7evXtbnz59wusPHDgQzt898sgj7ayzznI9t+qBXbNmjet9feaZZ9z2ZcuWuZ7p+vXrRzy20g2UZgEgvRDIAkA2KCdTl+BzSmkA4r90roDSr1SpUpk+htIMateubc8++6wLmhXIKoDdt2+f5QUFuB7lzIoXPGfHzp073b32W3mtfkWKFAn/rPQC5e4+/vjjNmHCBGvcuLG7eY+htosXL474HSlbtmy29wlAsJEjCwAJogFMGzZscHmpnk8//TSijXocxd8mukyWekCV+xrLb7/95gZb3X333S6PVs/pDQLzKEdXvJzaWI455hjXToPK/AG1BnspNzcvVKlSxQXeP/zwgzsZ8N+8AWpywQUX2J49e9ygNgWyCmw9GsCm16Ue4ejH0MA7AOmFHlkAyAZdwt64cWPEOg2AqlSpknXs2NFd8u7Vq5eNGjXKfv/9dzdQyk8Bl9IP7r33Xjfg6dtvv3UDqfwGDx7seiBvuOEGu+6661zAOXv2bJduoDQGXUIfN26cu8yvdII77rgj4vdVhUC9ugoEa9So4aoDRJfeUs/y9ddfbwMGDHCPqTSBkSNH2p9//uku/eeWUgKiA/Rjjz3Whg4d6npbtT8aoKX3U4PIFIxrAJu3b6quoEFgSnO47LLLwo+h91eBbc+ePd37psBWVQ0U+OsEoEuXLrnedwABkuwkXQAICg2W0n+b0bcGDRqE26xatSrUunXrUPHixUP169cPTZ8+/ZDBT/Pnzw81btw4VLJkyVCbNm1CkydPjhjsJXPmzAmddtppoRIlSoQqVKgQ6tSpU3iA2MyZM0ONGjVy25o0aeLaRj/Hs88+G6pZs2aocOHCoXbt2h0y2Es0YOvGG28MVapUyT2WBpgtXLgwvD2egWmxxHqPdPvoo4/cdg0wO+mkk9x7VLFixVDbtm1Db775ZsRjTJs2zf2OtkXToLQhQ4aE6tSp4wasVatWLXThhReGvvrqK7edwV5A+iikf5IdTAMAAADZRY4sAAAAAolAFgAAAIFEIAsAAIBAIpAFAABAIBHIAgAAIJAIZAEAABBIBLIAAAAIJAJZAAAABBKBLAAAAAKJQBYAAACBRCALAACAQCKQBQAAgAXR/wE4XishJHGb5gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 700x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Salary Distribution by Education Level\n",
    "plt.figure(figsize=(7,4))\n",
    "sns.boxplot(data=df, x='Education Level', y='Salary')\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.title('Salary Distribution by Education Level')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a6729585",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArEAAAGGCAYAAABsTdmlAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjMsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvZiW1igAAAAlwSFlzAAAPYQAAD2EBqD+naQAAeWBJREFUeJzt3Qd4U9X7B/C3ew9m2bvsvfceigMUB4gKiIAIiPATARURHCiogKAMFRD/KIIKKiJDNrKX7F02lNlJd/N/vqfcmLRpm7Rpm9t+P88Tktzc3Nzck9A3577nPU4Gg8EgREREREQ64pzXO0BEREREZCsGsURERESkOwxiiYiIiEh3GMQSERERke4wiCUiIiIi3WEQS0RERES6wyCWiIiIiHSHQSwRERER6Q6DWCIiIiLSHQaxRFQgbd68WZycnNQ1FRyhoaHy1FNPSZEiRVT7z5gxQ/KjCxcuqPe3aNGivN4VohzDIJbIgXz11VfqD0+zZs3yelccTnx8vMycOVMaNGgg/v7+EhgYKLVq1ZLBgwfLyZMn83r3dOnhhx+WQoUKqcAutfDwcClZsqT6LCYnJ0t+MWrUKFm7dq2MHz9evv/+e3nooYfSXRffxfQur7zySq7uNxGl5WphGRHlkSVLlkiFChVkz549cvbsWalSpUpe75LD6NWrl/z111/Sp08fGTRokCQkJKjgddWqVdKyZUupXr16Xu+iLn801a5dWwV2P/zwg9ljb731lty+fVvWrFkjzs75p79j48aN0qNHD3njjTesWr9Lly7y4osvplletWpVcWTly5eXmJgYcXNzy+tdIcoxDGKJHERISIjs2LFDfv31VxkyZIgKaCdOnJir+4AeN/R4enp6iiPZu3evClY//PBDFVyZmj17toSFhUlei46OFh8fH9GTihUrqs/Y2LFjpX///tK1a1fj8Z47d64K9OrVq5fj+xEbGyvu7u65EizfvHlT9eJbC8Hq888/L3qRmJiovsc4no72PSayt/zz85pI5xC04tTuI488onL2cF+DXsfChQvLgAED0jwvIiJC/bEy7VmKi4tTwQl6cj08PKRs2bLy5ptvquWmcFp0+PDh6rVwah7roucNPv30U9XDidxBLy8vadSokfz8889pXh+9Pa+99poULVpU/Pz85PHHH5erV6+qbb/33ntm62L5Sy+9JEFBQeq18JoL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+sL20peEXGwv7cy1roPYTp06yZEjR8yWYWAV8l7Hjh2rAkIcyA0bNqjSWnDq1ClVUqtFixbqPq4//PBDFQyjvBagERBEYoCWts7q1avNXgfraNtAygIGjeF1evbsqZYlJyer+xgElh6U68IlNeyzvT8AObFNyjlsL31he+kL20tf2F764pYL7WXt9h06iPXz85PatWubLfPx8VE1YbXlAwcOVL2dhQsXVoHpiBEjVPDZvHlz9Th6PRGsvvDCCzJ16lSV//rOO++owWJagPnKK6+ont0333xTXnrpJdm4caMsW7ZM/vzzT+Pr4jX69esnjRs3lqZNm8qMGTMkOjpaBdVERERElLscOoi1BspgoVIAemIxiApVBb766ivj40gDWLVqlQwdOlQFtwiCEYxOnjzZuA7KayFgRc3ZmTNnSpkyZeSbb75R29I8++yzcuvWLVVfFoEwynWtWbMmzWAvIiIiIsp5ugtiMbOWKQz4Qs1XXNJTvnz5NOkCqbVv314OHjyY4TpIHcgofYCIiIiIcocu6sQSEREREZliEEtEREREusMgloiIiIh0R3c5sXpmMBhsqn9mbd02FAXGNlmixPGxvfSF7aUvbC99YXvpS0IutpcWJ2lxU3oYxOaiyMhIdY36tkRERESUcdwUEBCQ7uNOhszCXLIbTJBw7do1Vf/WycnJLtvUZgHDFL32mgWMcg7bS1/YXvrC9tIXtpe+RORieyE0RQBbqlQpVUY1PeyJzUVoCNSgzQn4QPE/Af1ge+kL20tf2F76wvbSF/9caq+MemA1HNhFRERERLrDIJaIiIiIdIdBrM55eHjIxIkT1TU5PraXvrC99IXtpS9sL33xcMD24sAuIiIiItId9sQSERERke4wiCUiIiIi3WEQS0RERES6wyBW57788kupUKGCeHp6SrNmzWTPnj15vUskIlu3bpXHHntMFWrGxBYrV640exyp6O+++66ULFlSvLy8pHPnznLmzJk829+CbMqUKdKkSRM1CUnx4sWlZ8+ecurUKbN1YmNjZdiwYVKkSBHx9fWVXr16SWhoaJ7tc0E3Z84cqVu3rrFeZYsWLeSvv/4yPs72clwff/yx+j/x9ddfNy5jezmW9957T7WR6aV69eoO2V4MYnXsp59+ktGjR6vRggcOHJB69epJt27d5ObNm3m9awVedHS0ag/8yLBk6tSp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",
      "text/plain": [
       "<Figure size 700x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Experience vs Salary Line Chart\n",
    "# Group by experience and calculate mean salary\n",
    "experience_salary = df.groupby(\"Years of Experience\")[\"Salary\"].mean().reset_index()\n",
    "# Plot the line chart\n",
    "plt.figure(figsize=(7,4))\n",
    "sns.lineplot(data=experience_salary, x=\"Years of Experience\", y=\"Salary\", marker='o')\n",
    "plt.title(\"Average Salary vs. Years of Experience\")\n",
    "plt.xlabel(\"Years of Experience\")\n",
    "plt.ylabel(\"Average Salary\")\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1f87727e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split into train and test sets\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b9ec5647",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train a model (Random Forest)\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "model = RandomForestRegressor()\n",
    "model.fit(X_train, Y_train)\n",
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f8af35b9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Linear Regression - MAE: 31529.09538761926, MSE: 1640008783.6351976, R2: 0.25698266339874587\n",
      "Decision Tree Regressor - MAE: 25012.12042942468, MSE: 1236297435.86852, R2: 0.43988688523370856\n",
      "Random Forest Regressor - MAE: 24808.348100465908, MSE: 1206800428.386007, R2: 0.4532507087424973\n",
      "KNN Regressor - MAE: 26633.439453125, MSE: 1379699200.0, R2: 0.37491780519485474\n"
     ]
    }
   ],
   "source": [
    "# Load the models for evaluation\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"Decision Tree Regressor\": DecisionTreeRegressor(),\n",
    "    \"Random Forest Regressor\": RandomForestRegressor(),\n",
    "    \"KNN Regressor\": KNeighborsRegressor()\n",
    "}\n",
    "\n",
    "# Train and evaluate each model\n",
    "results = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, Y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    MAE = mean_absolute_error(Y_test, y_pred)\n",
    "    MSE = mean_squared_error(Y_test, y_pred)\n",
    "    R2 = r2_score(Y_test, y_pred)\n",
    "    \n",
    "    results[name] = {\"MAE\": MAE, \"MSE\": MSE, \"R2\": R2}\n",
    "    print(f\"{name} - MAE: {MAE}, MSE: {MSE}, R2: {R2}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "20447f54",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Best Model Based on R² Score:\n",
      "KNN Regressor with R² = 0.3749\n"
     ]
    }
   ],
   "source": [
    "# Find the best model based on R² score\n",
    "print(\"\\nBest Model Based on R² Score:\")\n",
    "print(f\"{name} with R² = {R2:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0c592b9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predict salary for a new person (example)\n",
    "X_new = np.array([[\"United States\", \"Master’s degree\", 15]])\n",
    "X_new[:, 0] = le_country.transform(X_new[:, 0])\n",
    "X_new[:, 1] = le_education.transform(X_new[:, 1])\n",
    "X_new = X_new.astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c12bf058",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Salary (USD): 176200.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hillo\\AppData\\Roaming\\Python\\Python313\\site-packages\\sklearn\\utils\\validation.py:2749: UserWarning: X does not have valid feature names, but KNeighborsRegressor was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "predicted_salary = model.predict(X_new)\n",
    "print(\"Predicted Salary (USD):\", round(predicted_salary[0], 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ca79b284",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['le_education.pkl']"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Save the model and encoders using joblib\n",
    "import joblib\n",
    "\n",
    "# Save model\n",
    "joblib.dump(model, \"salary_model.pkl\", compress=3)\n",
    "\n",
    "# Save encoders\n",
    "joblib.dump(le_country, \"le_country.pkl\", compress=3)\n",
    "joblib.dump(le_education, \"le_education.pkl\", compress=3)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "6da4e027",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting app.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile app.py\n",
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load model and encoders\n",
    "model = joblib.load(\"salary_model.pkl\")\n",
    "le_country = joblib.load(\"le_country.pkl\")\n",
    "le_education = joblib.load(\"le_education.pkl\")\n",
    "\n",
    "# App config\n",
    "st.set_page_config(page_title=\"Salary Predictor\", page_icon=\"💼\", layout=\"centered\")\n",
    "\n",
    "# App title\n",
    "st.title(\"Employee Salary Prediction\")\n",
    "st.markdown(\"<h5>Enter your details to estimate the expected salary:</h5>\", unsafe_allow_html=True)\n",
    "\n",
    "# Input form\n",
    "with st.form(\"salary_form\"):\n",
    "    col1, col2 = st.columns(2)\n",
    "\n",
    "    with col1:\n",
    "        country = st.selectbox(\" **🌍 Country** \", le_country.classes_)\n",
    "    with col2:\n",
    "        education = st.selectbox(\" **🎓 Education Level** \", le_education.classes_)\n",
    "\n",
    "    experience = st.slider(\" **👨‍💼 Years of Experience** \", 0, 50, 3)\n",
    "\n",
    "    submitted = st.form_submit_button(\" **Predict Salary** \")\n",
    "\n",
    "    if submitted:\n",
    "        # Encode input\n",
    "        country_encoded = le_country.transform([country])[0]\n",
    "        education_encoded = le_education.transform([education])[0]\n",
    "        X_new = np.array([[country_encoded, education_encoded, experience]])\n",
    "\n",
    "        # Predict salary\n",
    "        predicted_salary = model.predict(X_new)[0]\n",
    "        st.success(f\" **Estimated Salary:** ${round(predicted_salary, 2):,}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "bb2fb5ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "# python -m streamlit run app.py"
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
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
