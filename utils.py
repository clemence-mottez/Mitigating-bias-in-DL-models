
import seaborn as sns


def get_colors(n=2):
    colors = ['#C6DCEC',  # Light Blue
        '#FFDEC2',  # Light Peach
        '#C9E6C9',  # Light Green
        '#F9E2AF',  # Pale Yellow
        '#E5C1CD',  # Light Pink
        '#D1E8E2',  # Light Teal
        '#FFE4B5',  # Soft Light Orange
        '#ECD5E3',  # Soft Lavender
        '#CCE5FF',  # Soft Blue
        '#FFD1BA']  # Light Coral
    return colors[:n]

def add_labels(ax, total_count=None, rotation=0):
    for p in ax.patches:
        height = p.get_height()
        if total_count is not None:
            percentage = (height / total_count) * 100
            label = f'{int(height)}\n({percentage:.1f}%)'
        else:
            label = f'{int(height)}'
        ax.annotate(label, 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='baseline', fontsize=12, color='black', 
                    xytext=(0, -15), textcoords='offset points', rotation=rotation)

def plot_categorical(df, column, ax, colors, total_count=None, rotation=0):
    sns.countplot(x=column, data=df, hue=column, ax=ax, palette=colors)
    ax.set_title(column.capitalize())
    if rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right")
    add_labels(ax, total_count)

def plot_age_distribution(df, ax, hue=None, palette=None):
    if hue:
        sns.kdeplot(data=df, x='age', hue=hue, fill=True, ax=ax, palette=palette, alpha=0.5, legend=False)
        ax.set_title(f'Age Density by {hue.capitalize()}')
    else:
        sns.kdeplot(df['age'], fill=True, ax=ax, color='C3')
        ax.set_title('Age (All)')

def plot_age_boxplot(df, ax, x=None, hue=None, palette=None, order=None):
    sns.boxplot(data=df, x=x, y='age', hue=hue, palette=palette, ax=ax, order=order)
    ax.set_title(f'Age Distribution by {x.capitalize()}' if x else 'Age (All)')
    ax.set_ylim(15, 100)
    if x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

def plot_insurance_percentage(df, ax, group_by, hue_order=None, palette=None):
    df_percent = df.groupby([group_by, 'insurance_type']).size().reset_index(name='count')
    percentage = df_percent.groupby(group_by)['count'].apply(lambda x: 100 * x / x.sum())
    df_percent["percentage"] = percentage.reset_index(level=0, drop=True)
    sns.barplot(x='insurance_type', y='percentage', hue=group_by, data=df_percent, ax=ax, hue_order=hue_order, palette=palette)
    ax.set_title(f'Health Insurance Percentage by {group_by.capitalize()}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
