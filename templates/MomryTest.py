import pandas as pd

# حساب الذاكرة قبل التحسين
bf = df.memory_usage(deep=True).sum()

# تحسين أنواع البيانات
df = (
    df
    # تحويل الأعمدة الزمنية
    .assign(**{
        col: pd.to_datetime(df[col], errors='coerce')
        for col in ['order_date', 'first_login_date', 'first_order_date']
    })
    # تحويل الأعمدة الرقمية إلى int32
    .assign(order_id=pd.to_numeric(df['order_id'], downcast='integer'),
            ops_center_id=pd.to_numeric(df['ops_center_id'], downcast='integer'),
            total_customer_orders=pd.to_numeric(df['total_customer_orders'], downcast='integer'))
    # معالجة الأعمدة النصية المتكررة
    .astype({col: 'category' for col in df.select_dtypes(include='object').columns})
    # معالجة الأعمدة المنطقية
    .assign(vip_user_order_flg=df['vip_user_order_flg'].map({'t': True, 'f': False}).astype(bool),
            first_order_low_nps=df['first_order_low_nps'].map({'t': True, 'f': False}).astype(bool),
            express_delivery_flg=df['express_delivery_flg'].map({'t': True, 'f': False}).astype(bool),
            is_vip_user=df['is_vip_user'].map({'t': True, 'f': False}).astype(bool))
    # تعيين 'order_id' كفهرس
    .set_index('order_id')
)

# حساب الذاكرة بعد التحسين
af = df.memory_usage(deep=True).sum()
memory_saved = bf - af
memory_saved_percentage = (memory_saved / bf) * 100

# طباعة النتائج
print(f'Memory before conversion: {bf} bytes')
print(f'Memory after conversion: {af} bytes')
print(f'Memory saved: {memory_saved} bytes')
print(f'Memory saved percentage: {memory_saved_percentage:.2f}%')
