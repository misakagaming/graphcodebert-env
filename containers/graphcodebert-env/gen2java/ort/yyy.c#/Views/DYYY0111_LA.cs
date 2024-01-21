// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: DYYY0111_LA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:51
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF LOCAL VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: DYYY0111_LA
  /// </summary>
  [Serializable]
  public class DYYY0111_LA : ViewBase, ILocalView
  {
    private static DYYY0111_LA[] freeArray = new DYYY0111_LA[30];
    private static int countFree = 0;
    
    // Entity View: LOC
    //        Type: DONT_CHANGE_RETURN_CODES
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    private char _LocDontChangeReturnCodesQ1Ok_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    public char LocDontChangeReturnCodesQ1Ok_AS {
      get {
        return(_LocDontChangeReturnCodesQ1Ok_AS);
      }
      set {
        _LocDontChangeReturnCodesQ1Ok_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesQ1Ok
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesQ1Ok;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesQ1Ok
    /// </summary>
    public int LocDontChangeReturnCodesQ1Ok {
      get {
        return(_LocDontChangeReturnCodesQ1Ok);
      }
      set {
        _LocDontChangeReturnCodesQ1Ok = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesN40ObjCreateFailed
    /// </summary>
    private char _LocDontChangeReturnCodesN40ObjCreateFailed_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesN40ObjCreateFailed
    /// </summary>
    public char LocDontChangeReturnCodesN40ObjCreateFailed_AS {
      get {
        return(_LocDontChangeReturnCodesN40ObjCreateFailed_AS);
      }
      set {
        _LocDontChangeReturnCodesN40ObjCreateFailed_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesN40ObjCreateFailed
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesN40ObjCreateFailed;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesN40ObjCreateFailed
    /// </summary>
    public int LocDontChangeReturnCodesN40ObjCreateFailed {
      get {
        return(_LocDontChangeReturnCodesN40ObjCreateFailed);
      }
      set {
        _LocDontChangeReturnCodesN40ObjCreateFailed = value;
      }
    }
    // Entity View: LOC
    //        Type: DONT_CHANGE_REASON_CODES
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    private char _LocDontChangeReasonCodesQ1Default_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    public char LocDontChangeReasonCodesQ1Default_AS {
      get {
        return(_LocDontChangeReasonCodesQ1Default_AS);
      }
      set {
        _LocDontChangeReasonCodesQ1Default_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ1Default
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ1Default;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ1Default
    /// </summary>
    public int LocDontChangeReasonCodesQ1Default {
      get {
        return(_LocDontChangeReasonCodesQ1Default);
      }
      set {
        _LocDontChangeReasonCodesQ1Default = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ101ParentNotFound
    /// </summary>
    private char _LocDontChangeReasonCodesQ101ParentNotFound_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ101ParentNotFound
    /// </summary>
    public char LocDontChangeReasonCodesQ101ParentNotFound_AS {
      get {
        return(_LocDontChangeReasonCodesQ101ParentNotFound_AS);
      }
      set {
        _LocDontChangeReasonCodesQ101ParentNotFound_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ101ParentNotFound
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ101ParentNotFound;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ101ParentNotFound
    /// </summary>
    public int LocDontChangeReasonCodesQ101ParentNotFound {
      get {
        return(_LocDontChangeReasonCodesQ101ParentNotFound);
      }
      set {
        _LocDontChangeReasonCodesQ101ParentNotFound = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ102ParentAlreadyExist
    /// </summary>
    private char _LocDontChangeReasonCodesQ102ParentAlreadyExist_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ102ParentAlreadyExist
    /// </summary>
    public char LocDontChangeReasonCodesQ102ParentAlreadyExist_AS {
      get {
        return(_LocDontChangeReasonCodesQ102ParentAlreadyExist_AS);
      }
      set {
        _LocDontChangeReasonCodesQ102ParentAlreadyExist_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ102ParentAlreadyExist
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ102ParentAlreadyExist;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ102ParentAlreadyExist
    /// </summary>
    public int LocDontChangeReasonCodesQ102ParentAlreadyExist {
      get {
        return(_LocDontChangeReasonCodesQ102ParentAlreadyExist);
      }
      set {
        _LocDontChangeReasonCodesQ102ParentAlreadyExist = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ103ParentAttrValueInvalid
    /// </summary>
    private char _LocDontChangeReasonCodesQ103ParentAttrValueInvalid_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ103ParentAttrValueInvalid
    /// </summary>
    public char LocDontChangeReasonCodesQ103ParentAttrValueInvalid_AS {
      get {
        return(_LocDontChangeReasonCodesQ103ParentAttrValueInvalid_AS);
      }
      set {
        _LocDontChangeReasonCodesQ103ParentAttrValueInvalid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ103ParentAttrValueInvalid
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ103ParentAttrValueInvalid;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ103ParentAttrValueInvalid
    /// </summary>
    public int LocDontChangeReasonCodesQ103ParentAttrValueInvalid {
      get {
        return(_LocDontChangeReasonCodesQ103ParentAttrValueInvalid);
      }
      set {
        _LocDontChangeReasonCodesQ103ParentAttrValueInvalid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ111ParentsTypeNotFound
    /// </summary>
    private char _LocDontChangeReasonCodesQ111ParentsTypeNotFound_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ111ParentsTypeNotFound
    /// </summary>
    public char LocDontChangeReasonCodesQ111ParentsTypeNotFound_AS {
      get {
        return(_LocDontChangeReasonCodesQ111ParentsTypeNotFound_AS);
      }
      set {
        _LocDontChangeReasonCodesQ111ParentsTypeNotFound_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ111ParentsTypeNotFound
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ111ParentsTypeNotFound;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ111ParentsTypeNotFound
    /// </summary>
    public int LocDontChangeReasonCodesQ111ParentsTypeNotFound {
      get {
        return(_LocDontChangeReasonCodesQ111ParentsTypeNotFound);
      }
      set {
        _LocDontChangeReasonCodesQ111ParentsTypeNotFound = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public DYYY0111_LA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public DYYY0111_LA( DYYY0111_LA orig )
    {
      LocDontChangeReturnCodesQ1Ok_AS = orig.LocDontChangeReturnCodesQ1Ok_AS;
      LocDontChangeReturnCodesQ1Ok = orig.LocDontChangeReturnCodesQ1Ok;
      LocDontChangeReturnCodesN40ObjCreateFailed_AS = orig.LocDontChangeReturnCodesN40ObjCreateFailed_AS;
      LocDontChangeReturnCodesN40ObjCreateFailed = orig.LocDontChangeReturnCodesN40ObjCreateFailed;
      LocDontChangeReasonCodesQ1Default_AS = orig.LocDontChangeReasonCodesQ1Default_AS;
      LocDontChangeReasonCodesQ1Default = orig.LocDontChangeReasonCodesQ1Default;
      LocDontChangeReasonCodesQ101ParentNotFound_AS = orig.LocDontChangeReasonCodesQ101ParentNotFound_AS;
      LocDontChangeReasonCodesQ101ParentNotFound = orig.LocDontChangeReasonCodesQ101ParentNotFound;
      LocDontChangeReasonCodesQ102ParentAlreadyExist_AS = orig.LocDontChangeReasonCodesQ102ParentAlreadyExist_AS;
      LocDontChangeReasonCodesQ102ParentAlreadyExist = orig.LocDontChangeReasonCodesQ102ParentAlreadyExist;
      LocDontChangeReasonCodesQ103ParentAttrValueInvalid_AS = orig.LocDontChangeReasonCodesQ103ParentAttrValueInvalid_AS;
      LocDontChangeReasonCodesQ103ParentAttrValueInvalid = orig.LocDontChangeReasonCodesQ103ParentAttrValueInvalid;
      LocDontChangeReasonCodesQ111ParentsTypeNotFound_AS = orig.LocDontChangeReasonCodesQ111ParentsTypeNotFound_AS;
      LocDontChangeReasonCodesQ111ParentsTypeNotFound = orig.LocDontChangeReasonCodesQ111ParentsTypeNotFound;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static DYYY0111_LA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new DYYY0111_LA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new DYYY0111_LA());
          }
          else 
          {
            DYYY0111_LA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new DYYY0111_LA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      LocDontChangeReturnCodesQ1Ok_AS = ' ';
      LocDontChangeReturnCodesQ1Ok = 0;
      LocDontChangeReturnCodesN40ObjCreateFailed_AS = ' ';
      LocDontChangeReturnCodesN40ObjCreateFailed = 0;
      LocDontChangeReasonCodesQ1Default_AS = ' ';
      LocDontChangeReasonCodesQ1Default = 0;
      LocDontChangeReasonCodesQ101ParentNotFound_AS = ' ';
      LocDontChangeReasonCodesQ101ParentNotFound = 0;
      LocDontChangeReasonCodesQ102ParentAlreadyExist_AS = ' ';
      LocDontChangeReasonCodesQ102ParentAlreadyExist = 0;
      LocDontChangeReasonCodesQ103ParentAttrValueInvalid_AS = ' ';
      LocDontChangeReasonCodesQ103ParentAttrValueInvalid = 0;
      LocDontChangeReasonCodesQ111ParentsTypeNotFound_AS = ' ';
      LocDontChangeReasonCodesQ111ParentsTypeNotFound = 0;
    }
  }
}