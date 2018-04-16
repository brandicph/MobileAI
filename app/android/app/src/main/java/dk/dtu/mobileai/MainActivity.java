package dk.dtu.mobileai;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Paint;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.telephony.CellInfo;
import android.telephony.CellInfoLte;
import android.telephony.PhoneStateListener;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.GridLabelRenderer;
import com.jjoe64.graphview.LegendRenderer;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static android.Manifest.permission.ACCESS_FINE_LOCATION;
import static android.Manifest.permission.READ_PHONE_STATE;


public class MainActivity extends AppCompatActivity {

    public static final int REQUEST_ID_MULTIPLE_PERMISSIONS = 124;

    private static final String TAG = "MainActivity";

    public static DataStore mDataStore = DataStore.getInstance();


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //setup content stuff
        this.setContentView(R.layout.activity_main);

        //setStatusBarColor(findViewById(R.id.statusBarBackground),getResources().getColor(android.R.color.white));


        // Adding if condition inside button.

        // If All permission is enabled successfully then this block will execute.
        if(CheckingPermissionIsEnabledOrNot())
        {
            Toast.makeText(MainActivity.this, "All Permissions Granted Successfully", Toast.LENGTH_LONG).show();
            try {
                mDataStore.init(this);
            } catch (SecurityException e){
                RequestMultiplePermission();
            } catch (Exception e) {
                Log.e(TAG, e.getMessage());

            }
        } else {

            //Calling method to enable permission.
            RequestMultiplePermission();

        }

    }

    public void setStatusBarColor(View statusBar, int color){
        Window w = getWindow();
        w.setFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS,WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        //status bar height
        int actionBarHeight = getActionBarHeight();
        int statusBarHeight = getStatusBarHeight();
        //action bar height
        statusBar.getLayoutParams().height = actionBarHeight + statusBarHeight;
        statusBar.setBackgroundColor(color);
    }

    public int getActionBarHeight() {
        int actionBarHeight = 0;
        TypedValue tv = new TypedValue();
        if (getTheme().resolveAttribute(android.R.attr.actionBarSize, tv, true))
        {
            actionBarHeight = TypedValue.complexToDimensionPixelSize(tv.data,getResources().getDisplayMetrics());
        }
        return actionBarHeight;
    }

    public int getStatusBarHeight() {
        int result = 0;
        int resourceId = getResources().getIdentifier("status_bar_height", "dimen", "android");
        if (resourceId > 0) {
            result = getResources().getDimensionPixelSize(resourceId);
        }
        return result;
    }

    public void onResume() {
        super.onResume();
        /*
        mTimer = new Runnable() {
            @Override
            public void run() {
                graphLastXValue += 0.25d;
                mSeries.appendData(new DataPoint(graphLastXValue, getRandom()), true, 22);
                mHandler.postDelayed(this, 330);
            }
        };
        mHandler.postDelayed(mTimer, 1500);
        */
    }

    @Override
    public void onPause() {
        super.onPause();

        try{
            //if(signalStrengthListener != null){tm.listen(signalStrengthListener, SignalStrengthListener.LISTEN_NONE);}
        }catch(Exception e){
            e.printStackTrace();
        }

    }


    public void onDestroy() {
        super.onDestroy();

        try{
            //if(signalStrengthListener != null){tm.listen(signalStrengthListener, SignalStrengthListener.LISTEN_NONE);}
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    //Permission function starts from here
    private void RequestMultiplePermission() {

        // Creating String Array with Permissions.
        ActivityCompat.requestPermissions(MainActivity.this, new String[]
                {
                        ACCESS_FINE_LOCATION,
                        READ_PHONE_STATE
                }, REQUEST_ID_MULTIPLE_PERMISSIONS);

    }

    // Calling override method.
    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {

            case REQUEST_ID_MULTIPLE_PERMISSIONS:

                if (grantResults.length > 0) {

                    boolean AccessFineLocation = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean ReadPhoneState = grantResults[1] == PackageManager.PERMISSION_GRANTED;

                    if (AccessFineLocation && ReadPhoneState) {

                        Toast.makeText(MainActivity.this, "Permission Granted", Toast.LENGTH_LONG).show();
                    }
                    else {
                        Toast.makeText(MainActivity.this,"Permission Denied",Toast.LENGTH_LONG).show();

                    }
                }

                break;
        }
    }

    // Checking permission is enabled or not using function starts from here.
    public boolean CheckingPermissionIsEnabledOrNot() {

        int FirstPermissionResult = ContextCompat.checkSelfPermission(getApplicationContext(), ACCESS_FINE_LOCATION);
        int SecondPermissionResult = ContextCompat.checkSelfPermission(getApplicationContext(), READ_PHONE_STATE);

        return FirstPermissionResult == PackageManager.PERMISSION_GRANTED &&
                SecondPermissionResult == PackageManager.PERMISSION_GRANTED ;
    }

}

