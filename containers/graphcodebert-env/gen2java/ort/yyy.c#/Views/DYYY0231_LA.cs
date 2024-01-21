// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: DYYY0231_LA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:06
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
  /// Internal data view storage for: DYYY0231_LA
  /// </summary>
  [Serializable]
  public class DYYY0231_LA : ViewBase, ILocalView
  {
    private static DYYY0231_LA[] freeArray = new DYYY0231_LA[30];
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
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesN10ObjNotFound
    /// </summary>
    private char _LocDontChangeReturnCodesN10ObjNotFound_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesN10ObjNotFound
    /// </summary>
    public char LocDontChangeReturnCodesN10ObjNotFound_AS {
      get {
        return(_LocDontChangeReturnCodesN10ObjNotFound_AS);
      }
      set {
        _LocDontChangeReturnCodesN10ObjNotFound_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesN10ObjNotFound
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesN10ObjNotFound;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesN10ObjNotFound
    /// </summary>
    public int LocDontChangeReturnCodesN10ObjNotFound {
      get {
        return(_LocDontChangeReturnCodesN10ObjNotFound);
      }
      set {
        _LocDontChangeReturnCodesN10ObjNotFound = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReturnCodesN41ObjUpdateFailed
    /// </summary>
    private char _LocDontChangeReturnCodesN41ObjUpdateFailed_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReturnCodesN41ObjUpdateFailed
    /// </summary>
    public char LocDontChangeReturnCodesN41ObjUpdateFailed_AS {
      get {
        return(_LocDontChangeReturnCodesN41ObjUpdateFailed_AS);
      }
      set {
        _LocDontChangeReturnCodesN41ObjUpdateFailed_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReturnCodesN41ObjUpdateFailed
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReturnCodesN41ObjUpdateFailed;
    /// <summary>
    /// Attribute for: LocDontChangeReturnCodesN41ObjUpdateFailed
    /// </summary>
    public int LocDontChangeReturnCodesN41ObjUpdateFailed {
      get {
        return(_LocDontChangeReturnCodesN41ObjUpdateFailed);
      }
      set {
        _LocDontChangeReturnCodesN41ObjUpdateFailed = value;
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
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ121ChildNotFound
    /// </summary>
    private char _LocDontChangeReasonCodesQ121ChildNotFound_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ121ChildNotFound
    /// </summary>
    public char LocDontChangeReasonCodesQ121ChildNotFound_AS {
      get {
        return(_LocDontChangeReasonCodesQ121ChildNotFound_AS);
      }
      set {
        _LocDontChangeReasonCodesQ121ChildNotFound_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ121ChildNotFound
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ121ChildNotFound;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ121ChildNotFound
    /// </summary>
    public int LocDontChangeReasonCodesQ121ChildNotFound {
      get {
        return(_LocDontChangeReasonCodesQ121ChildNotFound);
      }
      set {
        _LocDontChangeReasonCodesQ121ChildNotFound = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ122ChildAlreadyExist
    /// </summary>
    private char _LocDontChangeReasonCodesQ122ChildAlreadyExist_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ122ChildAlreadyExist
    /// </summary>
    public char LocDontChangeReasonCodesQ122ChildAlreadyExist_AS {
      get {
        return(_LocDontChangeReasonCodesQ122ChildAlreadyExist_AS);
      }
      set {
        _LocDontChangeReasonCodesQ122ChildAlreadyExist_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ122ChildAlreadyExist
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ122ChildAlreadyExist;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ122ChildAlreadyExist
    /// </summary>
    public int LocDontChangeReasonCodesQ122ChildAlreadyExist {
      get {
        return(_LocDontChangeReasonCodesQ122ChildAlreadyExist);
      }
      set {
        _LocDontChangeReasonCodesQ122ChildAlreadyExist = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ123ChildAttrValueInvalid
    /// </summary>
    private char _LocDontChangeReasonCodesQ123ChildAttrValueInvalid_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ123ChildAttrValueInvalid
    /// </summary>
    public char LocDontChangeReasonCodesQ123ChildAttrValueInvalid_AS {
      get {
        return(_LocDontChangeReasonCodesQ123ChildAttrValueInvalid_AS);
      }
      set {
        _LocDontChangeReasonCodesQ123ChildAttrValueInvalid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ123ChildAttrValueInvalid
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ123ChildAttrValueInvalid;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ123ChildAttrValueInvalid
    /// </summary>
    public int LocDontChangeReasonCodesQ123ChildAttrValueInvalid {
      get {
        return(_LocDontChangeReasonCodesQ123ChildAttrValueInvalid);
      }
      set {
        _LocDontChangeReasonCodesQ123ChildAttrValueInvalid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// </summary>
    private char _LocDontChangeReasonCodesQ132ChildConcurrencyError_AS;
    /// <summary>
    /// Attribute missing flag for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// </summary>
    public char LocDontChangeReasonCodesQ132ChildConcurrencyError_AS {
      get {
        return(_LocDontChangeReasonCodesQ132ChildConcurrencyError_AS);
      }
      set {
        _LocDontChangeReasonCodesQ132ChildConcurrencyError_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _LocDontChangeReasonCodesQ132ChildConcurrencyError;
    /// <summary>
    /// Attribute for: LocDontChangeReasonCodesQ132ChildConcurrencyError
    /// </summary>
    public int LocDontChangeReasonCodesQ132ChildConcurrencyError {
      get {
        return(_LocDontChangeReasonCodesQ132ChildConcurrencyError);
      }
      set {
        _LocDontChangeReasonCodesQ132ChildConcurrencyError = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public DYYY0231_LA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public DYYY0231_LA( DYYY0231_LA orig )
    {
      LocDontChangeReturnCodesQ1Ok_AS = orig.LocDontChangeReturnCodesQ1Ok_AS;
      LocDontChangeReturnCodesQ1Ok = orig.LocDontChangeReturnCodesQ1Ok;
      LocDontChangeReturnCodesN10ObjNotFound_AS = orig.LocDontChangeReturnCodesN10ObjNotFound_AS;
      LocDontChangeReturnCodesN10ObjNotFound = orig.LocDontChangeReturnCodesN10ObjNotFound;
      LocDontChangeReturnCodesN41ObjUpdateFailed_AS = orig.LocDontChangeReturnCodesN41ObjUpdateFailed_AS;
      LocDontChangeReturnCodesN41ObjUpdateFailed = orig.LocDontChangeReturnCodesN41ObjUpdateFailed;
      LocDontChangeReasonCodesQ1Default_AS = orig.LocDontChangeReasonCodesQ1Default_AS;
      LocDontChangeReasonCodesQ1Default = orig.LocDontChangeReasonCodesQ1Default;
      LocDontChangeReasonCodesQ121ChildNotFound_AS = orig.LocDontChangeReasonCodesQ121ChildNotFound_AS;
      LocDontChangeReasonCodesQ121ChildNotFound = orig.LocDontChangeReasonCodesQ121ChildNotFound;
      LocDontChangeReasonCodesQ122ChildAlreadyExist_AS = orig.LocDontChangeReasonCodesQ122ChildAlreadyExist_AS;
      LocDontChangeReasonCodesQ122ChildAlreadyExist = orig.LocDontChangeReasonCodesQ122ChildAlreadyExist;
      LocDontChangeReasonCodesQ123ChildAttrValueInvalid_AS = orig.LocDontChangeReasonCodesQ123ChildAttrValueInvalid_AS;
      LocDontChangeReasonCodesQ123ChildAttrValueInvalid = orig.LocDontChangeReasonCodesQ123ChildAttrValueInvalid;
      LocDontChangeReasonCodesQ132ChildConcurrencyError_AS = orig.LocDontChangeReasonCodesQ132ChildConcurrencyError_AS;
      LocDontChangeReasonCodesQ132ChildConcurrencyError = orig.LocDontChangeReasonCodesQ132ChildConcurrencyError;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static DYYY0231_LA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new DYYY0231_LA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new DYYY0231_LA());
          }
          else 
          {
            DYYY0231_LA result = freeArray[--countFree];
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
      return(new DYYY0231_LA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      LocDontChangeReturnCodesQ1Ok_AS = ' ';
      LocDontChangeReturnCodesQ1Ok = 0;
      LocDontChangeReturnCodesN10ObjNotFound_AS = ' ';
      LocDontChangeReturnCodesN10ObjNotFound = 0;
      LocDontChangeReturnCodesN41ObjUpdateFailed_AS = ' ';
      LocDontChangeReturnCodesN41ObjUpdateFailed = 0;
      LocDontChangeReasonCodesQ1Default_AS = ' ';
      LocDontChangeReasonCodesQ1Default = 0;
      LocDontChangeReasonCodesQ121ChildNotFound_AS = ' ';
      LocDontChangeReasonCodesQ121ChildNotFound = 0;
      LocDontChangeReasonCodesQ122ChildAlreadyExist_AS = ' ';
      LocDontChangeReasonCodesQ122ChildAlreadyExist = 0;
      LocDontChangeReasonCodesQ123ChildAttrValueInvalid_AS = ' ';
      LocDontChangeReasonCodesQ123ChildAttrValueInvalid = 0;
      LocDontChangeReasonCodesQ132ChildConcurrencyError_AS = ' ';
      LocDontChangeReasonCodesQ132ChildConcurrencyError = 0;
    }
  }
}