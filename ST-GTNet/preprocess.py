import pandas as pd


def read_pre_excel(file_path, year = 2030):

    df = pd.read_excel(file_path)
    if '耕地破碎化指数' in df.columns: df.drop(columns=['耕地破碎化指数'], inplace=True)
    if '耕地LPI' in df.columns: df.drop(columns=['耕地LPI'], inplace=True)
    df.rename(columns={'文件名':'城市', '耕地LPI(%)':'耕地LPI'}, inplace=True)
    df["年份"] = year

    coord_data = pd.read_excel('../gnnwr/data/x/coord.xlsx')
    merged_df = pd.merge(
        df,
        coord_data,
        on="城市",
        how="left"  # 保留所有主数据记录
    )
    merged_df.dropna(axis=0, how='any', inplace=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df['建设用地均斑'] *= 1e-5
    merged_df['耕地LPI'] *= 1e-2
    merged_df['耕地聚集指数'] *= 1e-2
    merged_df.to_excel(f'./excel/landuse/{year}sds.xlsx')


def main():
    read_pre_excel('./excel/landuse/sds_2060.xlsx', 2060)

if __name__ == '__main__':
    main()