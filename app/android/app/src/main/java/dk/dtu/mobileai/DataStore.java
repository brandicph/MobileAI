package dk.dtu.mobileai;

import android.content.Context;

import java.io.IOException;

import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class DataStore {

    private static Context mContext;

    private static final DataStore ourInstance = new DataStore();

    public static DataStore getInstance() {
        return ourInstance;
    }


    private static InfoModule mInfoModule;
    private static LocationModule mLocationModule;
    private static CellularModule mCellularModule;

    // Define the interceptor, add authentication headers
    private static Interceptor interceptor = new Interceptor() {
        @Override
        public okhttp3.Response intercept(Chain chain) throws IOException {
            Request newRequest = chain.request()
                    .newBuilder()
                    .addHeader("Authorization", "Token 94629c425a5fd26dc172e3916e168975b648d80c")
                    .addHeader("Content-Type", "application/json")
                    .build();
            return chain.proceed(newRequest);
        }
    };

    // Add the interceptor to OkHttpClient
    private static OkHttpClient.Builder builder = new OkHttpClient.Builder().addInterceptor(interceptor);
    private static OkHttpClient client = builder.build();

    public static final String BASE_URL = "http://192.168.1.139:8000/";
    public static final Retrofit retrofit = new Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .client(client)
            .build();

    private DataStore() {}

    public void init(Context context){
        mContext = context;

        mInfoModule = new InfoModule(mContext);
        mLocationModule = new LocationModule(mContext);
        mCellularModule = new CellularModule(mContext);
    }

}
