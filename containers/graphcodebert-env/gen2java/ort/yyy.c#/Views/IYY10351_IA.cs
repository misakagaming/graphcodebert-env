// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: IYY10351_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:36
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
// START OF IMPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: IYY10351_IA
  /// </summary>
  [Serializable]
  public class IYY10351_IA : ViewBase, IImportView
  {
    private static IYY10351_IA[] freeArray = new IYY10351_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP_FILTER
    //        Type: IYY1_LIST
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListSortOption
    /// </summary>
    private char _ImpFilterIyy1ListSortOption_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListSortOption
    /// </summary>
    public char ImpFilterIyy1ListSortOption_AS {
      get {
        return(_ImpFilterIyy1ListSortOption_AS);
      }
      set {
        _ImpFilterIyy1ListSortOption_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListSortOption
    /// Domain: Text
    /// Length: 3
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListSortOption;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListSortOption
    /// </summary>
    public string ImpFilterIyy1ListSortOption {
      get {
        return(_ImpFilterIyy1ListSortOption);
      }
      set {
        _ImpFilterIyy1ListSortOption = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListScrollType
    /// </summary>
    private char _ImpFilterIyy1ListScrollType_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListScrollType
    /// </summary>
    public char ImpFilterIyy1ListScrollType_AS {
      get {
        return(_ImpFilterIyy1ListScrollType_AS);
      }
      set {
        _ImpFilterIyy1ListScrollType_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListScrollType
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListScrollType;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListScrollType
    /// </summary>
    public string ImpFilterIyy1ListScrollType {
      get {
        return(_ImpFilterIyy1ListScrollType);
      }
      set {
        _ImpFilterIyy1ListScrollType = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListListDirection
    /// </summary>
    private char _ImpFilterIyy1ListListDirection_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListListDirection
    /// </summary>
    public char ImpFilterIyy1ListListDirection_AS {
      get {
        return(_ImpFilterIyy1ListListDirection_AS);
      }
      set {
        _ImpFilterIyy1ListListDirection_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListListDirection
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListListDirection;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListListDirection
    /// </summary>
    public string ImpFilterIyy1ListListDirection {
      get {
        return(_ImpFilterIyy1ListListDirection);
      }
      set {
        _ImpFilterIyy1ListListDirection = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    private char _ImpFilterIyy1ListScrollAmount_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    public char ImpFilterIyy1ListScrollAmount_AS {
      get {
        return(_ImpFilterIyy1ListScrollAmount_AS);
      }
      set {
        _ImpFilterIyy1ListScrollAmount_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListScrollAmount
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFilterIyy1ListScrollAmount;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    public int ImpFilterIyy1ListScrollAmount {
      get {
        return(_ImpFilterIyy1ListScrollAmount);
      }
      set {
        _ImpFilterIyy1ListScrollAmount = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    private char _ImpFilterIyy1ListOrderByFieldNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    public char ImpFilterIyy1ListOrderByFieldNum_AS {
      get {
        return(_ImpFilterIyy1ListOrderByFieldNum_AS);
      }
      set {
        _ImpFilterIyy1ListOrderByFieldNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListOrderByFieldNum
    /// Domain: Number
    /// Length: 1
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private short _ImpFilterIyy1ListOrderByFieldNum;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    public short ImpFilterIyy1ListOrderByFieldNum {
      get {
        return(_ImpFilterIyy1ListOrderByFieldNum);
      }
      set {
        _ImpFilterIyy1ListOrderByFieldNum = value;
      }
    }
    // Entity View: IMP_FROM
    //        Type: IYY1_TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromIyy1TypeTinstanceId
    /// </summary>
    private char _ImpFromIyy1TypeTinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromIyy1TypeTinstanceId
    /// </summary>
    public char ImpFromIyy1TypeTinstanceId_AS {
      get {
        return(_ImpFromIyy1TypeTinstanceId_AS);
      }
      set {
        _ImpFromIyy1TypeTinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromIyy1TypeTinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpFromIyy1TypeTinstanceId;
    /// <summary>
    /// Attribute for: ImpFromIyy1TypeTinstanceId
    /// </summary>
    public string ImpFromIyy1TypeTinstanceId {
      get {
        return(_ImpFromIyy1TypeTinstanceId);
      }
      set {
        _ImpFromIyy1TypeTinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromIyy1TypeTkeyAttrText
    /// </summary>
    private char _ImpFromIyy1TypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromIyy1TypeTkeyAttrText
    /// </summary>
    public char ImpFromIyy1TypeTkeyAttrText_AS {
      get {
        return(_ImpFromIyy1TypeTkeyAttrText_AS);
      }
      set {
        _ImpFromIyy1TypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromIyy1TypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpFromIyy1TypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFromIyy1TypeTkeyAttrText
    /// </summary>
    public string ImpFromIyy1TypeTkeyAttrText {
      get {
        return(_ImpFromIyy1TypeTkeyAttrText);
      }
      set {
        _ImpFromIyy1TypeTkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromIyy1TypeTsearchAttrText
    /// </summary>
    private char _ImpFromIyy1TypeTsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromIyy1TypeTsearchAttrText
    /// </summary>
    public char ImpFromIyy1TypeTsearchAttrText_AS {
      get {
        return(_ImpFromIyy1TypeTsearchAttrText_AS);
      }
      set {
        _ImpFromIyy1TypeTsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromIyy1TypeTsearchAttrText
    /// Domain: Text
    /// Length: 20
    /// Varying Length: N
    /// </summary>
    private string _ImpFromIyy1TypeTsearchAttrText;
    /// <summary>
    /// Attribute for: ImpFromIyy1TypeTsearchAttrText
    /// </summary>
    public string ImpFromIyy1TypeTsearchAttrText {
      get {
        return(_ImpFromIyy1TypeTsearchAttrText);
      }
      set {
        _ImpFromIyy1TypeTsearchAttrText = value;
      }
    }
    // Entity View: IMP_FILTER_START
    //        Type: IYY1_TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStartIyy1TypeTkeyAttrText
    /// </summary>
    private char _ImpFilterStartIyy1TypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStartIyy1TypeTkeyAttrText
    /// </summary>
    public char ImpFilterStartIyy1TypeTkeyAttrText_AS {
      get {
        return(_ImpFilterStartIyy1TypeTkeyAttrText_AS);
      }
      set {
        _ImpFilterStartIyy1TypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStartIyy1TypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterStartIyy1TypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFilterStartIyy1TypeTkeyAttrText
    /// </summary>
    public string ImpFilterStartIyy1TypeTkeyAttrText {
      get {
        return(_ImpFilterStartIyy1TypeTkeyAttrText);
      }
      set {
        _ImpFilterStartIyy1TypeTkeyAttrText = value;
      }
    }
    // Entity View: IMP_FILTER_STOP
    //        Type: IYY1_TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStopIyy1TypeTkeyAttrText
    /// </summary>
    private char _ImpFilterStopIyy1TypeTkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStopIyy1TypeTkeyAttrText
    /// </summary>
    public char ImpFilterStopIyy1TypeTkeyAttrText_AS {
      get {
        return(_ImpFilterStopIyy1TypeTkeyAttrText_AS);
      }
      set {
        _ImpFilterStopIyy1TypeTkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStopIyy1TypeTkeyAttrText
    /// Domain: Text
    /// Length: 4
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterStopIyy1TypeTkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFilterStopIyy1TypeTkeyAttrText
    /// </summary>
    public string ImpFilterStopIyy1TypeTkeyAttrText {
      get {
        return(_ImpFilterStopIyy1TypeTkeyAttrText);
      }
      set {
        _ImpFilterStopIyy1TypeTkeyAttrText = value;
      }
    }
    // Entity View: IMP_FILTER
    //        Type: IYY1_TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1TypeTsearchAttrText
    /// </summary>
    private char _ImpFilterIyy1TypeTsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1TypeTsearchAttrText
    /// </summary>
    public char ImpFilterIyy1TypeTsearchAttrText_AS {
      get {
        return(_ImpFilterIyy1TypeTsearchAttrText_AS);
      }
      set {
        _ImpFilterIyy1TypeTsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1TypeTsearchAttrText
    /// Domain: Text
    /// Length: 20
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1TypeTsearchAttrText;
    /// <summary>
    /// Attribute for: ImpFilterIyy1TypeTsearchAttrText
    /// </summary>
    public string ImpFilterIyy1TypeTsearchAttrText {
      get {
        return(_ImpFilterIyy1TypeTsearchAttrText);
      }
      set {
        _ImpFilterIyy1TypeTsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1TypeTotherAttrText
    /// </summary>
    private char _ImpFilterIyy1TypeTotherAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1TypeTotherAttrText
    /// </summary>
    public char ImpFilterIyy1TypeTotherAttrText_AS {
      get {
        return(_ImpFilterIyy1TypeTotherAttrText_AS);
      }
      set {
        _ImpFilterIyy1TypeTotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1TypeTotherAttrText
    /// Domain: Text
    /// Length: 2
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1TypeTotherAttrText;
    /// <summary>
    /// Attribute for: ImpFilterIyy1TypeTotherAttrText
    /// </summary>
    public string ImpFilterIyy1TypeTotherAttrText {
      get {
        return(_ImpFilterIyy1TypeTotherAttrText);
      }
      set {
        _ImpFilterIyy1TypeTotherAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public IYY10351_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public IYY10351_IA( IYY10351_IA orig )
    {
      ImpFilterIyy1ListSortOption_AS = orig.ImpFilterIyy1ListSortOption_AS;
      ImpFilterIyy1ListSortOption = orig.ImpFilterIyy1ListSortOption;
      ImpFilterIyy1ListScrollType_AS = orig.ImpFilterIyy1ListScrollType_AS;
      ImpFilterIyy1ListScrollType = orig.ImpFilterIyy1ListScrollType;
      ImpFilterIyy1ListListDirection_AS = orig.ImpFilterIyy1ListListDirection_AS;
      ImpFilterIyy1ListListDirection = orig.ImpFilterIyy1ListListDirection;
      ImpFilterIyy1ListScrollAmount_AS = orig.ImpFilterIyy1ListScrollAmount_AS;
      ImpFilterIyy1ListScrollAmount = orig.ImpFilterIyy1ListScrollAmount;
      ImpFilterIyy1ListOrderByFieldNum_AS = orig.ImpFilterIyy1ListOrderByFieldNum_AS;
      ImpFilterIyy1ListOrderByFieldNum = orig.ImpFilterIyy1ListOrderByFieldNum;
      ImpFromIyy1TypeTinstanceId_AS = orig.ImpFromIyy1TypeTinstanceId_AS;
      ImpFromIyy1TypeTinstanceId = orig.ImpFromIyy1TypeTinstanceId;
      ImpFromIyy1TypeTkeyAttrText_AS = orig.ImpFromIyy1TypeTkeyAttrText_AS;
      ImpFromIyy1TypeTkeyAttrText = orig.ImpFromIyy1TypeTkeyAttrText;
      ImpFromIyy1TypeTsearchAttrText_AS = orig.ImpFromIyy1TypeTsearchAttrText_AS;
      ImpFromIyy1TypeTsearchAttrText = orig.ImpFromIyy1TypeTsearchAttrText;
      ImpFilterStartIyy1TypeTkeyAttrText_AS = orig.ImpFilterStartIyy1TypeTkeyAttrText_AS;
      ImpFilterStartIyy1TypeTkeyAttrText = orig.ImpFilterStartIyy1TypeTkeyAttrText;
      ImpFilterStopIyy1TypeTkeyAttrText_AS = orig.ImpFilterStopIyy1TypeTkeyAttrText_AS;
      ImpFilterStopIyy1TypeTkeyAttrText = orig.ImpFilterStopIyy1TypeTkeyAttrText;
      ImpFilterIyy1TypeTsearchAttrText_AS = orig.ImpFilterIyy1TypeTsearchAttrText_AS;
      ImpFilterIyy1TypeTsearchAttrText = orig.ImpFilterIyy1TypeTsearchAttrText;
      ImpFilterIyy1TypeTotherAttrText_AS = orig.ImpFilterIyy1TypeTotherAttrText_AS;
      ImpFilterIyy1TypeTotherAttrText = orig.ImpFilterIyy1TypeTotherAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static IYY10351_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new IYY10351_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new IYY10351_IA());
          }
          else 
          {
            IYY10351_IA result = freeArray[--countFree];
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
      return(new IYY10351_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpFilterIyy1ListSortOption_AS = ' ';
      ImpFilterIyy1ListSortOption = "   ";
      ImpFilterIyy1ListScrollType_AS = ' ';
      ImpFilterIyy1ListScrollType = " ";
      ImpFilterIyy1ListListDirection_AS = ' ';
      ImpFilterIyy1ListListDirection = " ";
      ImpFilterIyy1ListScrollAmount_AS = ' ';
      ImpFilterIyy1ListScrollAmount = 0;
      ImpFilterIyy1ListOrderByFieldNum_AS = ' ';
      ImpFilterIyy1ListOrderByFieldNum = 0;
      ImpFromIyy1TypeTinstanceId_AS = ' ';
      ImpFromIyy1TypeTinstanceId = "00000000000000000000";
      ImpFromIyy1TypeTkeyAttrText_AS = ' ';
      ImpFromIyy1TypeTkeyAttrText = "    ";
      ImpFromIyy1TypeTsearchAttrText_AS = ' ';
      ImpFromIyy1TypeTsearchAttrText = "                    ";
      ImpFilterStartIyy1TypeTkeyAttrText_AS = ' ';
      ImpFilterStartIyy1TypeTkeyAttrText = "    ";
      ImpFilterStopIyy1TypeTkeyAttrText_AS = ' ';
      ImpFilterStopIyy1TypeTkeyAttrText = "    ";
      ImpFilterIyy1TypeTsearchAttrText_AS = ' ';
      ImpFilterIyy1TypeTsearchAttrText = "                    ";
      ImpFilterIyy1TypeTotherAttrText_AS = ' ';
      ImpFilterIyy1TypeTotherAttrText = "  ";
    }
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IImportView orig )
    {
      this.CopyFrom((IYY10351_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IYY10351_IA orig )
    {
      ImpFilterIyy1ListSortOption_AS = orig.ImpFilterIyy1ListSortOption_AS;
      ImpFilterIyy1ListSortOption = orig.ImpFilterIyy1ListSortOption;
      ImpFilterIyy1ListScrollType_AS = orig.ImpFilterIyy1ListScrollType_AS;
      ImpFilterIyy1ListScrollType = orig.ImpFilterIyy1ListScrollType;
      ImpFilterIyy1ListListDirection_AS = orig.ImpFilterIyy1ListListDirection_AS;
      ImpFilterIyy1ListListDirection = orig.ImpFilterIyy1ListListDirection;
      ImpFilterIyy1ListScrollAmount_AS = orig.ImpFilterIyy1ListScrollAmount_AS;
      ImpFilterIyy1ListScrollAmount = orig.ImpFilterIyy1ListScrollAmount;
      ImpFilterIyy1ListOrderByFieldNum_AS = orig.ImpFilterIyy1ListOrderByFieldNum_AS;
      ImpFilterIyy1ListOrderByFieldNum = orig.ImpFilterIyy1ListOrderByFieldNum;
      ImpFromIyy1TypeTinstanceId_AS = orig.ImpFromIyy1TypeTinstanceId_AS;
      ImpFromIyy1TypeTinstanceId = orig.ImpFromIyy1TypeTinstanceId;
      ImpFromIyy1TypeTkeyAttrText_AS = orig.ImpFromIyy1TypeTkeyAttrText_AS;
      ImpFromIyy1TypeTkeyAttrText = orig.ImpFromIyy1TypeTkeyAttrText;
      ImpFromIyy1TypeTsearchAttrText_AS = orig.ImpFromIyy1TypeTsearchAttrText_AS;
      ImpFromIyy1TypeTsearchAttrText = orig.ImpFromIyy1TypeTsearchAttrText;
      ImpFilterStartIyy1TypeTkeyAttrText_AS = orig.ImpFilterStartIyy1TypeTkeyAttrText_AS;
      ImpFilterStartIyy1TypeTkeyAttrText = orig.ImpFilterStartIyy1TypeTkeyAttrText;
      ImpFilterStopIyy1TypeTkeyAttrText_AS = orig.ImpFilterStopIyy1TypeTkeyAttrText_AS;
      ImpFilterStopIyy1TypeTkeyAttrText = orig.ImpFilterStopIyy1TypeTkeyAttrText;
      ImpFilterIyy1TypeTsearchAttrText_AS = orig.ImpFilterIyy1TypeTsearchAttrText_AS;
      ImpFilterIyy1TypeTsearchAttrText = orig.ImpFilterIyy1TypeTsearchAttrText;
      ImpFilterIyy1TypeTotherAttrText_AS = orig.ImpFilterIyy1TypeTotherAttrText_AS;
      ImpFilterIyy1TypeTotherAttrText = orig.ImpFilterIyy1TypeTotherAttrText;
    }
  }
}
