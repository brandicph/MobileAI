package dk.dtu.mobileai;

import android.content.Context;
import android.util.Log;

public class InfoModule {

    private static final String TAG = "InfoModule";

    private static Context mContext;

    public InfoModule(Context context){
        mContext = context;

        //Log.d(TAG, getDeviceName());
    }

    public String getDeviceName() {
        String manufacturer = android.os.Build.MANUFACTURER;
        String model = android.os.Build.MODEL;
        if (model.startsWith(manufacturer)) {
            return capitalize(model);
        } else {
            return capitalize(manufacturer) + " " + model;
        }
    }


    private String capitalize(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        char first = s.charAt(0);
        if (Character.isUpperCase(first)) {
            return s;
        } else {
            return Character.toUpperCase(first) + s.substring(1);
        }
    }
}
