package dk.dtu.mobileai;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.SystemClock;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;

import java.util.concurrent.TimeUnit;

import static android.Manifest.permission.ACCESS_FINE_LOCATION;
import static android.Manifest.permission.READ_PHONE_STATE;

public class SplashActivity extends AppCompatActivity {

    public static final int REQUEST_ID_MULTIPLE_PERMISSIONS = 124;

    private static final String TAG = "SplashActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //SystemClock.sleep(TimeUnit.SECONDS.toMillis(3));

        if(CheckingPermissionIsEnabledOrNot())
        {
            //Toast.makeText(TabbedActivity.this, "All Permissions Granted Successfully", Toast.LENGTH_LONG).show();
            //Snackbar.make(findViewById(R.id.main_content), "All Permissions Granted Successfully", Snackbar.LENGTH_LONG).setAction("Action", null).show();
            try {
                Intent intent = new Intent(this, TabbedActivity.class);
                startActivity(intent);
                finish();

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


    //Permission function starts from here
    private void RequestMultiplePermission() {

        // Creating String Array with Permissions.
        ActivityCompat.requestPermissions(SplashActivity.this, new String[]
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

                        //Toast.makeText(TabbedActivity.this, "Permission Granted", Toast.LENGTH_LONG).show();
                        Intent intent = new Intent(this, TabbedActivity.class);
                        startActivity(intent);
                        finish();

                    }
                    else {
                        Toast.makeText(SplashActivity.this,"Permission Denied",Toast.LENGTH_LONG).show();
                        //Snackbar.make(findViewById(R.id.main_content), "Permissions Denied", Snackbar.LENGTH_LONG).setAction("Action", null).show();

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
