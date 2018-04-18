package dk.dtu.mobileai.data;

import android.content.Context;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;

import java.io.IOException;

import dk.dtu.mobileai.R;
import dk.dtu.mobileai.TabbedActivity;
import dk.dtu.mobileai.modules.CellularModule;
import dk.dtu.mobileai.modules.InfoModule;
import dk.dtu.mobileai.modules.LocationModule;
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
    private static Interceptor interceptor;

    // Add the interceptor to OkHttpClient
    private static OkHttpClient.Builder builder;
    private static OkHttpClient client;

    public static Retrofit retrofit;

    private DataStore() {}

    public void init(Context context){
        mContext = context;

        setupModules();
        setupRetrofit();
    }

    public void setupModules(){
        mInfoModule = new InfoModule(mContext);
        mLocationModule = new LocationModule(mContext);
        mCellularModule = new CellularModule(mContext);
    }

    public void setupRetrofit(){
        interceptor = new Interceptor() {
            @Override
            public okhttp3.Response intercept(Chain chain) throws IOException {
                Request newRequest = chain.request()
                        .newBuilder()
                        .addHeader("Authorization", "Token " + getApiToken())
                        .addHeader("Content-Type", "application/json")
                        .build();
                return chain.proceed(newRequest);
            }
        };

        builder = new OkHttpClient.Builder().addInterceptor(interceptor);

        client = builder.build();

        retrofit =  new Retrofit.Builder()
                .baseUrl(getApiEndpoint())
                .addConverterFactory(GsonConverterFactory.create())
                .client(client)
                .build();
    }

    public String getApiEndpoint(){
        SharedPreferences settings = PreferenceManager.getDefaultSharedPreferences(mContext);
        String api_endpoint = settings.getString("api_endpoint", mContext.getString(R.string.pref_default_api_endpoint));
        return api_endpoint;
    }

    public String getApiToken(){
        SharedPreferences settings = PreferenceManager.getDefaultSharedPreferences(mContext);
        String api_token = settings.getString("api_token", mContext.getString(R.string.pref_default_api_token));
        return api_token;
    }

    public String getDeviceName(){
        return mCellularModule.getDeviceName();
    }

    public String getApiEntityId(){
        SharedPreferences settings = PreferenceManager.getDefaultSharedPreferences(mContext);
        String api_entity_id = settings.getString("api_entity_id", mContext.getString(R.string.pref_default_api_entity_id));
        return api_entity_id;
    }


    public boolean getApiSync(){
        SharedPreferences settings = PreferenceManager.getDefaultSharedPreferences(mContext);
        boolean api_sync = settings.getBoolean("api_sync", true);
        return api_sync;
    }
}
