package dk.dtu.mobileai.fragments;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ListView;

import java.util.ArrayList;

import dk.dtu.mobileai.adapters.NetworkAdapter;
import dk.dtu.mobileai.R;
import dk.dtu.mobileai.data.DataStore;
import dk.dtu.mobileai.models.NetworkItem;

/**
 * A placeholder fragment containing a simple view.
 */
public class NetworkFragment extends Fragment {

    private static NetworkAdapter mAdapter;
    private ArrayList<NetworkItem> arr = new ArrayList<NetworkItem>();

    public NetworkFragment() {
    }

    /**
     * Returns a new instance of this fragment for the given section
     * number.
     */
    public static NetworkFragment newInstance() {
        NetworkFragment fragment = new NetworkFragment();
        Bundle args = new Bundle();
        //args.putInt(ARG_SECTION_NUMBER, sectionNumber);
        fragment.setArguments(args);
        return fragment;
    }

    public void setData(ArrayList<NetworkItem> items){
        arr.clear();
        arr.addAll(items);

        if (mAdapter != null){
            mAdapter.notifyDataSetChanged();
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View rootView = inflater.inflate(R.layout.fragment_network, container, false);
        ListView listView = (ListView) rootView.findViewById(R.id.lvItems);


        DataStore dataStore = DataStore.getInstance();
        String deviceName = dataStore.getDeviceName();

        // Create the adapter to convert the array to views
        mAdapter = new NetworkAdapter(rootView.getContext(), arr);
        // Attach the adapter to a ListView
        listView.setAdapter(mAdapter);

        return rootView;
    }
}
